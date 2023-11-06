# Chrono imports
import pychrono as chrono
import pychrono.robot as robot_chrono
import cv2
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
import os

# Gymnasium imports
import gymnasium as gym


class cobra_corridor_mefloor(ChronoBaseEnv):
    """
    Wrapper for the cobra chrono model into a gym environment.
    Mainly built for use with action space = 
    """

    def __init__(self, render_mode='human'):
        ChronoBaseEnv.__init__(self, render_mode)

        SetChronoDataDirectories()

        # ----------------------------
        # Action and observation space
        # -----------------------------

        # Max steering in radians
        self.max_steer = np.pi / 6.
        # Max motor speed in radians per sec
        self.max_speed = 2*np.pi

        self.num_obs = np.random.randint(6, 12)
        # Define action space -> These will scale the max steer and max speed linearly
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float64)


        # -----------------------------
        # Chrono simulation parameters
        # -----------------------------
        self.system = None  # Chrono system set in reset method
        self.ground = None  # Ground body set in reset method
        self.rover = None  # Rover set in reset method

        self.x_obs = None
        self.y_obs = None

        self._initpos = chrono.ChVectorD(
            0.0, 0.0, 0.0)  # Rover initial position
        # Frequncy in which we apply control
        self._control_frequency = 10
        # Dynamics timestep
        self._step_size = 1e-3
        # Number of steps dynamics has to take before we apply control
        self._steps_per_control = round(
            1 / (self._step_size * self._control_frequency))
        self._collision = False
        self._terrain_length = 25
        self._terrain_width = 5
        self._terrain_height = 2
        self.vehicle_pos = None
        
        self.camera_width = 640
        self.camera_height = 320
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3,640,320), dtype=np.uint8)

        # ---------------------------------
        # Gym Environment variables
        # ---------------------------------
        # Maximum simulation time (seconds)
        self._max_time = 50
        # Holds reward of the episode
        self.reward = 0
        self._debug_reward = 0
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

        # Process exploration reward
        self.grid_resolution = 0.1
        grid_size = (int(self._terrain_length/self.grid_resolution)+1, int(self._terrain_width/self.grid_resolution)+1)
        self.grid = np.zeros(grid_size, dtype=int)
        self.prev_reward = 0.0
        
    def reset(self, seed=None, options=None):
        self._sens_manager = None
        self.cam = None
        self.x_obs =  None
        self.y_obs = None
        self.z_rot = None
        self.radius = None
        self.object_index = None
        self.object_radius = None
        self.prev_reward = 0.0
        
        """
        Reset the environment to its initial state -> Set up for standard gym API
        :param seed: Seed for the random number generator
        :param options: Options for the simulation (dictionary)
        """

        # -----------------------------
        # Set up system with collision
        # -----------------------------
        self.system = chrono.ChSystemNSC()
        self.system.Set_G_acc(chrono.ChVectorD(0, 0, -9.81))
        chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(0.0025)
        chrono.ChCollisionModel.SetDefaultSuggestedMargin(0.0025)

        # -----------------------------
        # Set up Terrain
        # -----------------------------
        ground_mat = chrono.ChMaterialSurfaceNSC()
        room_mmesh = chrono.ChTriangleMeshConnected()
        current_directory = os.getcwd()
        room_mmesh.LoadWavefrontMesh(
            current_directory+"/../envs/data/environment/hallway.obj", False, True)

        room_contact_mesh = chrono.ChTriangleMeshConnected()
        room_contact_mesh.LoadWavefrontMesh(
            current_directory+"/../envs/data/environment/hallway.obj", False, True)

        room_trimesh_shape = chrono.ChTriangleMeshShape()
        room_trimesh_shape.SetMesh(room_mmesh)
        room_trimesh_shape.SetName("Hallway Mesh")
        room_trimesh_shape.SetMutable(False)

        room_mesh_body = chrono.ChBody()
        room_mesh_body.SetPos(chrono.ChVectorD(0, 0, -0.1))
        room_mesh_body.AddVisualShape(room_trimesh_shape)
        room_mesh_body.SetBodyFixed(True)
        room_mesh_body.GetCollisionModel().ClearModel()
        room_mesh_body.GetCollisionModel().AddTriangleMesh(
            ground_mat, room_contact_mesh, True, True)
        room_mesh_body.GetCollisionModel().BuildModel()
        room_mesh_body.SetCollide(True)

        self.system.Add(room_mesh_body)

        # -----------------------------
        # Create the COBRA
        # -----------------------------
        self.rover = robot_chrono.Cobra(
            self.system, robot_chrono.CobraWheelType_SimpleWheel)
        self.driver = robot_chrono.CobraSpeedDriver(
            1/self._control_frequency, 0.0)
        self.rover.SetDriver(self.driver)

        # Initialize position of robot randomly
        self.initialize_robot_pos(seed)

        # -----------------------------
        # Add obstacles
        # -----------------------------

        self.add_obstacles(seed)

        # -----------------------------
        # Add sensors
        # -----------------------------
        self.add_sensors()

        # -----------------------------
        # Get the intial observation
        # -----------------------------
        self.observation = self.get_observation()
        # self._old_distance = np.linalg.norm(self.observation[:3] - self.goal)
        # _vector_to_goal is a chrono vector
        self._debug_reward = 0

        self._terminated = False
        self._truncated = False

        return self.observation, {}

    def step(self, action):
        """
        Take a step in the environment - Frequency by default is 10 Hz.
        """

        # Linearly interpolate steer angle between pi/6 and pi/8
        steer_angle = action[0] * self.max_steer
        wheel_speed = action[1] * self.max_speed
        self.driver.SetSteering(steer_angle)  # Maybe we should ramp this steer
        # Wheel speed is ramped up to wheel_speed with a ramp time of 1/control_frequency
        self.driver.SetMotorSpeed(wheel_speed)
    

        for i in range(self._steps_per_control):
            self.rover.Update()
            self.system.DoStepDynamics(self._step_size)
            self.vehicle_pos = self.rover.GetChassis().GetPos()
            self._sens_manager.Update()
            grid_x_cur = self.rover.GetChassis().GetPos().x/self.grid_resolution
            grid_y_cur = self.rover.GetChassis().GetPos().y/self.grid_resolution
            self.grid[int(grid_x_cur), int(grid_y_cur)] = 1
            # print(self.rover.GetChassis().GetPos())

        # Get the observation
        self.observation = self.get_observation()

        # Get reward
        self.reward = self.get_reward()
        self._debug_reward += self.reward

        # Check if we are done
        self._is_terminated()
        self._is_truncated()

        return self.observation, self.reward, self._terminated, self._truncated, {}

    def render(self, mode):
        """
        Render the environment
        """

        # ------------------------------------------------------
        # Add visualization - only if we want to see "human" POV
        # ------------------------------------------------------
        if mode == 'human':
            if self._render_setup == False:
                self.vis = chronoirr.ChVisualSystemIrrlicht()
                self.vis.AttachSystem(self.system)
                self.vis.SetCameraVertical(chrono.CameraVerticalDir_Z)
                self.vis.SetWindowSize(1280, 720)
                self.vis.SetWindowTitle('Cobro RL playground')
                self.vis.Initialize()
                self.vis.AddSkyBox()
                self.vis.AddCamera(chrono.ChVectorD(
                    0.5, 2.0, 0.5), chrono.ChVectorD(5.0, 4.0, 0))
                self.vis.AddTypicalLights()
                self.vis.AddLightWithShadow(chrono.ChVectorD(
                    1.5, -2.5, 5.5), chrono.ChVectorD(0, 0, 0.5), 3, 4, 10, 40, 512)
                self._render_setup = True

            self.vis.BeginScene()
            self.vis.Render()
            self.vis.EndScene()

        elif mode == 'rgb_array':
            camera_buffer_RGBA8 = self.cam.GetMostRecentRGBA8Buffer()
            rgb_data = None
            if camera_buffer_RGBA8.HasData():
                rgb = camera_buffer_RGBA8.GetRGBA8Data()[:, :, 0:3]
            else:
                rgb = np.zeros((self.camera_width, self.camera_height, 3))
            return rgb 

        else:
            raise NotImplementedError

    def get_reward(self):
        """
        Get the reward for the current step
        """
        
        reward = np.sum(self.grid) - self.prev_reward
        reward = reward + 0.0
        self.prev_reward = np.sum(self.grid)
        
        return reward

    def _is_terminated(self):
        """
        Check if the environment is terminated
        """

        self._terminated = False

        # If we have exceeded the max time -> Terminate
        if self.system.GetChTime() > self._max_time:
            print('--------------------------------------------------------------')
            print('Time out')
            print('Initial position: ', self._initpos)
            self._debug_reward += self.reward
            print('Reward: ', self.reward)
            print('Accumulated Reward: ', self._debug_reward)
            print('--------------------------------------------------------------')
            self._terminated = True

    def _is_truncated(self):
        """
        Check if the environment is truncated
        """
        # If we have collided -> Truncate and give big negative reward
        self.check_collision()
        if self._collision:
            self.reward -= 300
            self._debug_reward += self.reward
            self._truncated = True
            print('--------------------------------------------------------------')
            print('Crashed')
            print('Vehicle Postion: ', self.vehicle_pos)
            print('Accumulated Reward: ', self._debug_reward)
            print('--------------------------------------------------------------')
        # Vehicle should not fall off the terrain
        elif (abs(self.vehicle_pos.x) < 1.0 or (abs(self.vehicle_pos.x) > self._terrain_length - 1.0) or (abs(self.vehicle_pos.y) < 1.0) or (abs(self.vehicle_pos.y) > self._terrain_width - 1.0)):
            self.reward -= 300
            self._debug_reward += self.reward
            self._truncated = True
            print('--------------------------------------------------------------')
            print('Outside of terrain')
            print('Vehicle Position: ', self.vehicle_pos)
            print('Accumulated Reward: ', self._debug_reward)
            print('--------------------------------------------------------------')

    def initialize_robot_pos(self, seed=1):
        """
        Initialize the pose of the robot
        """
        self._initpos = chrono.ChVectorD(1.5, 1.9, 0.2)

        # For now no randomness
        self.rover.Initialize(chrono.ChFrameD(
            self._initpos, chrono.ChQuaternionD(1, 0, 0, 0)))

        self.vehicle_pos = self._initpos


    def get_observation(self):
        """
        Get the observation from the environment
        """
        camera_buffer_RGBA8 = self.cam.GetMostRecentRGBA8Buffer()
        rgb_data = None
        if camera_buffer_RGBA8.HasData():
            rgb_data = camera_buffer_RGBA8.GetRGBA8Data()[:, :, 0:3]
            rgb_data = np.transpose(rgb_data, (2, 1, 0)).astype(np.uint8)
        else:
            rgb_data = np.zeros((3, self.camera_width, self.camera_height)).astype(np.uint8)
        # For not just the priveledged position of the rover
        return rgb_data

    # -------------Add Random Objects to the environment -------------------------------------

    def add_obstacles(self, seed=1):
        """
        Add random obstacles to the environment
        """
        # np.random.seed(seed)

        self.x_obs = np.zeros(self.num_obs)
        self.y_obs = np.zeros(self.num_obs)
        self.z_rot = np.zeros(self.num_obs)
        self.radius = np.zeros(self.num_obs)
        self.object_index = np.zeros(self.num_obs)
        self.object_radius = np.zeros(self.num_obs)
        
        # Generate a random float 
        for i in range(self.num_obs):
            self.object_index[i] = np.random.randint(1, 3)
            
            if self.object_index[i] == 1:
                self.object_radius[i] = 0.25
            elif self.object_index[i] == 2:
                self.object_radius[i] = 0.4
                
            invalid = True
            while(invalid):
                x = 1.0 + ((self._terrain_length-2.0) - 0) * np.random.rand()
                y = 1.0 + ((self._terrain_width-2.0) - 0) * np.random.rand()
                if abs(x - self._initpos.x) > 1.0 or abs(y - self._initpos.y) > 1.0:
                    invalid = False
                for j in range(i):
                    if abs(x - self.x_obs[j]) < (self.object_radius[i] + self.object_radius[j]) and abs(y - self.y_obs[j]) < (self.object_radius[i] + self.object_radius[j]):
                        invalid = True
                        break
                    
                if invalid == False:
                    self.x_obs[i] = x
                    self.y_obs[i] = y
                    rot = 2 * np.pi * np.random.rand()
                    self.z_rot[i] = rot

            

        # Add obstacles to the environment
        for i in range(self.num_obs):
            # Generate index for random object
            
            mmesh_string = ""
            mmesh_radius = 0.0
            
            current_directory = os.getcwd()
            
            if(self.object_index[i] == 1):
                mmesh_string = current_directory+"/../envs/data/environment/obs/chair_1/swivel_chair_.obj"
            elif(self.object_index[i] == 2):
                mmesh_string = current_directory+"/../envs/data/environment/obs/scan_chair_1/textured.obj"

            
            self.radius[i] = self.object_radius[i]
            obstacle_mmesh = chrono.ChTriangleMeshConnected()
            obstacle_mmesh.LoadWavefrontMesh(mmesh_string, False, True)
    
            obstacle_trimesh_shape = chrono.ChTriangleMeshShape()
            obstacle_trimesh_shape.SetMesh(obstacle_mmesh)
            obstacle_trimesh_shape.SetName("obstacle")
            obstacle_trimesh_shape.SetMutable(False)
            
            obstacle_mesh_body = chrono.ChBody()
            obstacle_mesh_body.SetPos(chrono.ChVectorD(self.x_obs[i], self.y_obs[i], 0.0))
            obstacle_mesh_body.SetRot(chrono.Q_from_AngAxis(self.z_rot[i], chrono.ChVectorD(0, 0, 1)))
            obstacle_mesh_body.AddVisualShape(obstacle_trimesh_shape)
            obstacle_mesh_body.SetBodyFixed(True)

            self.system.Add(obstacle_mesh_body)

    # ------------------------------------- TODO: Add Sensors if necessary -------------------------------------

    def add_sensors(self):
        """
        Add sensors to the rover
        """

        self._sens_manager = sens.ChSensorManager(self.system)

        cam_offset_pose = chrono.ChFrameD(chrono.ChVectorD(0.18, 0, 0.35),
                                          chrono.Q_from_AngAxis(0, chrono.ChVectorD(0, 1, 0)))

        self.cam = sens.ChCameraSensor(self.rover.GetChassis().GetBody(), 30, cam_offset_pose, 640,  320, 1.408, 2)
        
        self.cam.PushFilter(sens.ChFilterRGBA8Access())
        self.cam.SetName("CameraSensor")
        self._sens_manager.AddSensor(self.cam)
        
        light_pos_1 = chrono.ChVectorD(4.0,2.0,1.85)
        intensity = 1.0
        self._sens_manager.scene.AddPointLight(chrono.ChVectorF(light_pos_1.x,light_pos_1.y,light_pos_1.z), chrono.ChColor(intensity, intensity, intensity), 500.0)

        light_pos_2 = chrono.ChVectorD(12.0,2.0,1.85)
        intensity = 1.0
        self._sens_manager.scene.AddPointLight(chrono.ChVectorF(light_pos_2.x,light_pos_2.y,light_pos_2.z), chrono.ChColor(intensity, intensity, intensity), 500.0)
        

    # ------------ Check for collision with objects -------------------------------------

    def check_collision(self):
        """
        Check if we collide with any of the objects
        """
        collide = False
        cur_pos = self.rover.GetChassis().GetPos()
        for i in range(self.num_obs):
            if abs(cur_pos.x - self.x_obs[i]) < 0.5+self.object_radius[i] and abs(cur_pos.y - self.y_obs[i]) < 0.5+self.object_radius[i]:
                collide = True
                break

        self._collision = collide
