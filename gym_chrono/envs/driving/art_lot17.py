# Chrono imports
import pychrono as chrono
import pychrono.vehicle as veh
import cmath
try:
    from pychrono import irrlicht as chronoirr
except:
    print('Could not import ChronoIrrlicht')
try:
    import pychrono.sensor as sens
except:
    print('Could not import Chrono Sensor')

try:
    from pychrono import irrlicht as chronoirr
except:
    print('Could not import ChronoIrrlicht')


# Gym chrono imports
# Custom imports
from gym_chrono.envs.ChronoBase import ChronoBaseEnv
from gym_chrono.envs.utils.utils import CalcInitialPose, chVector_to_npArray, npArray_to_chVector, SetChronoDataDirectories

# Standard Python imports
import os
import math
import numpy as np

# Gymnasium imports
import gymnasium as gym


class art_lot17(ChronoBaseEnv):
    """
    Gym environment for the ART vehicle Chrono simulation to reach a point in the lot 17 parking lot
    """

    def __init__(self, render_mode='human'):
        ChronoBaseEnv.__init__(self, render_mode)

        SetChronoDataDirectories()

        # Action space is the throttle and steering - Throttle is between 0 and 1, steering is -1 to 1
        self.action_space = gym.spaces.Box(
            low=[-1.0, 0], high=[1.0, 1.0], shape=(2,), dtype=np.float64)

        # Define observation space
        # First few elements describe the relative position of the rover to the goal
        # Delta x in local frame
        # Delta y in local frame
        # Vehicle heading
        # Heading needed to reach the goal
        # Velocity of vehicle
        self._num_observations = 5
        self.observation_space = gym.spaces.Box(
            low=-200, high=200, shape=(self._num_observations,), dtype=np.float64)

        # -----------------------------
        # Chrono simulation parameters
        # -----------------------------
        self.system = None  # Chrono system set in reset method
        self.vehicle = None  # Vehicle set in reset method
        self.ground = None  # Ground body set in reset method
        self.art = None  # ART set in reset method
        self.driver = None  # Driver set in reset method
        self.driver_input = None  # Driver input set in reset method

        self.x_obs = None
        self.y_obs = None

        self._initpos = chrono.ChVectorD(
            0.0, 0.0, 0.0)  # ART initial position

        # Frequncy in which we apply control
        self._control_frequency = 10
        # Dynamics timestep
        self._step_size = 1e-3
        # Number of steps dynamics has to take before we apply control
        self._steps_per_control = round(
            1 / (self._step_size * self._control_frequency))
        self._collision = False
        self.vehicle_pos = None

        self.sensor_manager = None
        self._have_gps = False  # Flag to check if GPS sensor is present
        self._have_imu = False  # Flag to check if IMU sensor is present
        self.gps_origin = None  # GPS origin in lat, long, alt
        self.goal_gps = None  # Goal in GPS frame
        self._sensor_frequency = 10  # Frequency of sensor frame update

        # -----------------------------
        # Terrain helper variables
        # -----------------------------
        self._terrain_length = 35
        self._terrain_width = 70
        self._terrain_height = 2
        self._terrain_center = chrono.ChVectorD(0, 0, 0)

        # If the vehicle is in contact with the wall, that is a collision
        self._wall_center = chrono.ChVectorD(0, 0, 0)
        self._wall_box_length = 15
        self._wall_box_width = 50

        # ---------------------------------
        # Gym Environment variables
        # ---------------------------------
        # Maximum simulation time (seconds)
        self._max_time = 100
        # Holds reward of the episode
        self.reward = 0
        self._debug_reward = 0
        # Position of goal as numpy array
        self.goal = None
        # Distance to goal at previos time step -> To gauge "progress"
        self._vector_to_goal = None
        self._old_distance = None
        # Observation of the environment
        self.observation = None
        # Flag to determine if the environment has terminated -> In the event of timeOut or reach goal
        self._terminated = False
        # Flag to determine if the environment has truncated -> In the event of a crash
        self._truncated = False
        # Flag to check if the render setup has been done -> Some problem if rendering is setup in reset
        self._render_setup = False
        # Flag to count success while testing
        self._success = False

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state -> Set up for standard gym API
        :param seed: Seed for the random number generator
        :param options: Options for the simulation (dictionary)
        """
        # Initialize the vehicle
        self.vehicle = veh.RCCar()

        # -----------------------------
        # Contact mand collision properties
        # -----------------------------
        contact_method = chrono.ChContactMethod_SMC
        self.vehicle.SetContactMethod(chrono.ChContactMethod_SMC)
        self.vehicle.SetChassisCollisionType(False)  # No collision for now

        # ---------------------------------
        # Initailize positon
        # ----------------------------------
        self.initialize_vehicle_pos(seed)

        # -----------------------------
        # Set vehicle properties
        # -----------------------------
        self.vehicle.SetChassisFixed(False)
        self.vehicle.SetTireType(veh.TireModelType_TMEASY)
        self.vehicle.SetTireStepSize(self._step_size)
        self.vehicle.SetMaxMotorVoltageRatio(0.16)
        self.vehicle.SetStallTorque(0.3)
        self.vehicle.SetTireRollingResistance(0.06)
        self.vehicle.Initialize()

        # ------------------
        # Visualizations
        # ------------------
        self.vehicle.SetChassisVisualizationType(
            veh.VisualizationType_PRIMITIVES)
        self.vehicle.SetWheelVisualizationType(
            veh.VisualizationType_PRIMITIVES)
        self.vehicle.SetSuspensionVisualizationType(
            veh.VisualizationType_PRIMITIVES)
        self.vehicle.SetSteeringVisualizationType(
            veh.VisualizationType_PRIMITIVES)
        self.vehicle.SetTireVisualizationType(veh.VisualizationType_PRIMITIVES)
        self.chassis_body = self.vehicle.GetChassisBody()

        # ---------------------------
        # Get chrono system of the vehicle
        # ---------------------------
        self.system = self.vehicle.GetVehicle().GetSystem()
        self.system.Set_G_acc(chrono.ChVectorD(0, 0, -9.81))

        # ---------------------------
        # Terrain
        # ---------------------------
        self.system = self.vehicle.GetVehicle().GetSystem()
        self.system.Set_G_acc(chrono.ChVectorD(0, 0, -9.81))

        self.terrain = veh.RigidTerrain(self.system)
        patch_mat = chrono.ChMaterialSurfaceSMC()
        patch_mat.SetFriction(0.9)
        patch_mat.SetRestitution(0.01)
        patch_mat.SetYoungModulus(2e7)

        # Just a flat terrain for now
        # patch = self.terrain.AddPatch(patch_mat, chrono.ChVectorD(-self.terrain_length/2, -self.terrain_width/2, 0), chrono.ChVectorD(
        #     0, 0, 1), self.terrain_length, self.terrain_width, self.terrain_thickness)
        patch = self.terrain.AddPatch(
            patch_mat, chrono.CSYSNORM, self.terrain_length, self.terrain_width)
        patch.SetTexture(self.chronopath +
                         'textures/concrete.jpg', 200, 200)
        patch.SetColor(chrono.ChColor(0.8, 0.8, 0.5))
        self.terrain.Initialize()

        # ---------------------------
        # Provide a delta to controls
        # ---------------------------
        # Set the time response for steering and throttle inputs.
        steering_time = 0.75
        # time to go from 0 to +1 (or from 0 to -1)
        throttle_time = .5
        # time to go from 0 to +1
        self.SteeringDelta = (self._step_size / steering_time)
        self.ThrottleDelta = (self._step_size / throttle_time)

        # ---------------------------
        # Add sensors
        # ---------------------------
        self.add_sensors()

        # ---------------------------
        # Get the driver system
        # ---------------------------
        self.driver = veh.ChDriver(self.vehicle.GetVehicle())
        self.driver_inputs = self.driver.GetInputs()

        # ---------------------------
        # Add obstacles - pass for now
        # ---------------------------
        self.add_obstacles(seed)

        # ---------------------------
        # Set up the goal
        # ---------------------------
        self.set_goalPoint(seed)

        # ---------------------------
        # Get the initial observation
        # ---------------------------
        self.observation = self.get_observation()
        self._old_distance = self._vector_to_goal.Length() # To track progress

        self._debug_reward = 0

        self._terminated = False
        self._truncated = False
        self._success = False # During testing phase to see number of successes

        return self.observation, {}

    def get_observation(self):
        """
        Get the observation of the environment
        Position of vehicle from GPS
        Position of goal in cartesian
        Vehicle heading priveledged information
        Vehicle velocity priveledged information

        :return: Observation of the environment
                 1. Delta x in local frame
                 2. Delta y in local frame
                 3. Vehicle heading
                 4. Heading needed to reach the goal
                 5. Velocity of vehicle
        """

        observation = np.zeroes(self._num_observations)

        gps_buffer = self.gps.GetMostRecentGPSBuffer()
        cur_gps_data = None
        self.vehicle_pos = self.chassis_body.GetPos()
        # Get the GPS data from the buffer
        if self._have_gps 
            if gps_buffer.HasData():
                cur_gps_data = gps_buffer.GetGPSData()
                cur_gps_data = chrono.ChVectorD(
                    cur_gps_data[1], cur_gps_data[0], cur_gps_data[2])
            else:
                cur_gps_data = chrono.ChVectorD(self.origin)
            # Position of vehicle in cartesian coodinates from the GPS buffer
            sens.GPS2Cartesian(cur_gps_data, self.gps_origin)
        else: # There is no gps, use previledged information
            cur_gps_data = self.vehicle_pos
            
        # Goal is currently not read from the GPS sensor
        self._vector_to_goal = npArray_to_chVector(self.goal) - cur_gps_data
        vector_to_goal_local = self.chassis_body.GetRot().RotateBack(self._vector_to_goal)

        ###### TODO: Use magnetometer here to get heading - need help from Nevindu/Harry
        # For now using priveledged information
        vehicle_heading = self.chassis_body.GetRot().Q_to_Euler123().z()
        ###### TODO: Use state estimator used in reality in simulation as well to get velocity - need help from Stefan/Ishaan
        # For now using priveldeged information
        vehicle_velocity = self.chassis_body.GetPos_dt()
        local_delX = vector_to_goal_local.x * \
            np.cos(vehicle_heading) + vector_to_goal_local.y * \
            np.sin(vehicle_heading)
        local_delY = -vector_to_goal_local.x * \
            np.sin(vehicle_heading) + vector_to_goal_local.y * \
            np.cos(vehicle_heading)
        target_heading_to_goal = np.arctan2(
            vector_to_goal_local.y, vector_to_goal_local.x)

        observation[0] = local_delX
        observation[1] = local_delY
        observation[2] = vehicle_heading
        observation[3] = target_heading_to_goal
        observation[4] = vehicle_velocity.Length()


        return observation

    def initialize_vehicle_pos(self, seed=1):
        """
        Initialize the pose of the robot
        """
        # Initialize vehicle at the left corner of the terrain
        # No randomness for now
        self._initpos = chrono.ChVectorD(0, -0.2, 0.08144073)
        self._initRot = chrono.ChQuaternionD(1, 0, 0, 0)

        self.vehicle.SetInitPosition(
            chrono.ChCoordsysD(self._initLoc, self._initRot))

        self.vehicle_pos = self._initpos

    def set_goalPoint(self, seed=None):
        """
        Set the goal point for the environment
        """
        # Set the goal point
        if seed is not None:
            np.random.seed(seed)

        # Select a random point in a rectange of dimension 32.5 X 65 centered at the origin
        # The point should not be within a rectangle of dimension 17.5 x 55 centered at the origin
        # The point should not be within 5 meters of where the vehicle starts

        wall_x_tolerance = 1.25
        wall_y_tolerance = 2.5

        goal_wall_x_max = self._wall_box_length/2 + wall_x_tolerance

        goal_wall_y_max = self._wall_box_width/2 + wall_y_tolerance

        boundary_x_tolerance = 2.5
        boundary_y_tolerance = 5

        goal_boundary_x_max = self._terrain_length/2 - boundary_x_tolerance

        goal_boundary_y_max = self._terrain_width/2 - boundary_y_tolerance

        vehicle_x_pos = self.vehicle_pos.x()
        vehicle_y_pos = self.vehicle_pos.y()

        self.goal = np.random.uniform(low=[-self._terrain_length/2, -self._terrain_width/2], high=[
            self._terrain_length/2, self._terrain_width/2], size=(2,))

        goal_is_inside_wall = np.abs(self.goal[0]) < goal_wall_x_max and np.abs(
            self.goal[1]) < goal_wall_y_max
        goal_is_outside_boundary = np.abs(self.goal[0]) > goal_boundary_x_max and np.abs(
            self.goal[1]) < goal_boundary_y_max
        goal_is_close_to_vehicle = (math.sqrt(
            (self.goal[0] - vehicle_x_pos)**2 + (self.goal[1] - vehicle_y_pos)**2) < 5)

        while (goal_is_inside_wall or goal_is_outside_boundary or goal_is_close_to_vehicle):
            self.goal = np.random.uniform(low=[-self._terrain_length/2, -self._terrain_width/2], high=[
                self._terrain_length/2, self._terrain_width/2], size=(2,))

        self.goal = np.append(self.goal, 0.08144073)

        print("Goal in cartesian frame is: ", self.goal)

        # -----------------------------
        # Set up goal visualization
        # -----------------------------
        goal_contact_material = chrono.ChMaterialSurfaceNSC()
        goal_mat = chrono.ChVisualMaterial()
        goal_mat.SetAmbientColor(chrono.ChColor(1., 0., 0.))
        goal_mat.SetDiffuseColor(chrono.ChColor(1., 0., 0.))

        goal_body = chrono.ChBodyEasySphere(
            0.2, 1000, True, False, goal_contact_material)

        goal_body.SetPos(chrono.ChVectorD(
            goal_pos[0], goal_pos[1], 0.2))
        goal_body.SetBodyFixed(True)
        goal_body.GetVisualShape(0).SetMaterial(0, goal_mat)

        self.system.Add(goal_body)

        # -----------------------------
        # Goal in the GPS frame
        # -----------------------------
        if (self._have_gps):
            self.goal_gps = npArray_to_chVector(self.goal)
            sens.Cartesian2GPS(self.gps_origin, self.goal_gps)

    def add_sensors(self):
        """
        Add sensors to the vehicle
        """

        self._initialize_sensor_manager()
        self._add_gps_sensor(std=0.05)
        self._have_gps = True
        self._add_magnetometer_sensor(std=0)
        self._have_imu = True

    def _initialize_sensor_manager(self):
        """
        Initializes chrono sensor manager
        """
        self.sensor_manager = sens.ChSensorManager(self.system)
        self.sensor_manager.scene.AddPointLight(
            chrono.ChVectorF(0, 0, 100), chrono.ChVectorF(1, 1, 1), 5000)
        b = sens.Background()
        b.color_horizon = chrono.ChVectorF(.6, .7, .8)
        b.color_zenith = chrono.ChVectorF(.4, .5, .6)
        b.mode = sens.BackgroundMode_GRADIENT
        self.sensor_manager.scene.SetBackground(b)

    def _add_gps_sensor(self, std):
        """
        Add a GPS sensor to the vehicle
        :param std: Standard deviation of the GPS sensor
        """
        if (self.sensor_manager is None):
            self.initialize_sensor_manager()

        noise_model = sens.ChNoiseNormal(chrono.ChVectorD(
            0, 0, 0), chrono.ChVectorD(std, std, std))
        gps_offset_pose = chrono.ChFrameD(chrono.ChVectorD(
            0, 0, 0), chrono.Q_from_AngAxis(0, chrono.ChVectorD(1, 0, 0)))

        # The lat long and altitude of the cartesian origin - This needs to be measured
        self.gps_origin = chrono.ChVectorD(43.073268, -89.400636, 260.0)

        gps = sens.ChGPSSensor(self.vehicle.GetChassisBody(), self._sensor_frequency,  # update rate
                               gps_offset_pose, self.gps_origin, noise_model)

        gps.SetName("GPS")
        gps.PushFilter(sens.ChFilterGPSAccess())
        self.sensor_manager.AddSensor(gps)

    def _add_magnetometer_sensor(self, std=0):
        """
        Add a magnetometer sensor to the vehicle
        :param std: Standard deviation of the magnetometer sensor
        """
        noise_model = sens.ChNoiseNormal(chrono.ChVectorD(
            0, 0, 0), chrono.ChVectorD(std, std, std))
        imu_offset_pose = chrono.ChFrameD(chrono.ChVectorD(
            0, 0, 0), chrono.Q_from_AngAxis(0, chrono.ChVectorD(1, 0, 0)))
        mag = sens.ChMagnetometerSensor(self.vehicle.GetChassisBody(
        ), 100, imu_offset_pose, noise_model, self.gps_origin)
        mag.SetName("IMU - Magnetometer")
        mag.PushFilter(sens.ChFilterMagnetAccess())
        self.sensor_manager.AddSensor(mag)

    def add_obstacles(seed):
        """
        Add obstacles to the environment
        """
        pass
