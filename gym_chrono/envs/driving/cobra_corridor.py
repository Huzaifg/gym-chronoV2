# Chrono imports
import pychrono as chrono
import pychrono.robot as robot_chrono
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
from gym_chrono.envs.utils.utils import CalcInitialPose, chVector_to_npArray, SetChronoDataDirectories

# Standard Python imports
import os
import math
import numpy as np

# Gymnasium imports
import gymnasium as gym


class cobra_corridor(ChronoBaseEnv):
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
        self.max_speed = np.pi

        # Define action space -> These will scale the max steer and max speed linearly
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float64)

        # Define observation space - For now this is just the x,y,z position of the robot
        self.observation_space = gym.spaces.Box(
            low=-20, high=20, shape=(3,), dtype=np.float64)

        # -----------------------------
        # Chrono simulation parameters
        # -----------------------------
        self.system = None  # Chrono system set in reset method
        self.ground = None  # Ground body set in reset method
        self.rover = None  # Rover set in reset method
        # Frequncy in which we apply control
        self._control_frequency = 10
        # Dynamics timestep
        self._step_size = 1e-3
        # Number of steps dynamics has to take before we apply control
        self._steps_per_control = round(
            1 / (self._step_size * self._control_frequency))
        self._collision = False
        self._terrain_length = 20
        self._terrain_width = 20
        self._terrain_height = 2

        # ---------------------------------
        # Gym Environment variables
        # ---------------------------------
        # Maximum simulation time (seconds)
        self._max_time = 50
        # Holds reward of the episode
        self.reward = 0
        # Position of goal as numpy array
        self.goal = None
        # Distance to goal at previos time step -> To gauge "progress"
        self._old_distance = None
        # Observation of the environment
        self.observation = None
        # Flag to determine if the environment has terminated -> In the event of timeOut or reach goal
        self._terminated = False
        # Flag to determine if the environment has truncated -> In the event of a crash
        self._truncated = False

    def reset(self, seed=None, options=None):
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
        self.ground = chrono.ChBodyEasyBox(
            self._terrain_length, self._terrain_width, self._terrain_height, 1000, True, True, ground_mat)
        self.ground.SetPos(chrono.ChVectorD(0, 0, -self._terrain_height / 2))
        self.ground.SetBodyFixed(True)
        self.ground.GetVisualShape(0).SetTexture(
            chrono.GetChronoDataFile('textures/concrete.jpg'), 200, 200)
        self.system.Add(self.ground)

        self.add_obstacles(seed)

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
        # Add sensors
        # -----------------------------
        self.add_sensors()

        # ------------------------------------------------------
        # Add visualization - only if we want to see "human" POV
        # ------------------------------------------------------

        if self.render_mode == 'human':
            self.vis = chronoirr.ChVisualSystemIrrlicht()
            self.vis.AttachSystem(self.system)
            self.vis.SetCameraVertical(chrono.CameraVerticalDir_Z)
            self.vis.SetWindowSize(1280, 720)
            self.vis.SetWindowTitle('Cobro RL playground')
            self.vis.Initialize()
            self.vis.AddSkyBox()
            self.vis.AddCamera(chrono.ChVectorD(
                0, 2.5, 1.5), chrono.ChVectorD(0, 0, 1))
            self.vis.AddTypicalLights()
            self.vis.AddLightWithShadow(chrono.ChVectorD(
                1.5, -2.5, 5.5), chrono.ChVectorD(0, 0, 0.5), 3, 4, 10, 40, 512)

        # -----------------------------
        # Get the intial observation
        # -----------------------------
        self.observation = self.get_observation()

        # ---------------------------------------
        # Set the goal point and set premilinaries
        # ---------------------------------------
        self.set_goalPoint(seed=1)
        self._old_distance = np.linalg.norm(self.observation - self.goal)

        return self.observation, {}

    def step(self, action):
        """
        Take a step in the environment - Frequency by default is 10 Hz.
        """
        steer_angle = action[0] * self.max_steer
        wheel_speed = action[1] * self.max_speed

        self.driver.SetSteering(steer_angle)  # Maybe we should ramp this steer
        # Wheel speed is ramped up to wheel_speed with a ramp time of 1/control_frequency
        self.driver.SetMotorSpeed(wheel_speed)

        for i in range(self._steps_per_control):
            self.rover.Update()
            self.system.DoStepDynamics(self._step_size)

        # Get the observation
        self.observation = self.get_observation()
        # Get reward
        self.reward = self.get_reward()
        # Check if we are done
        self._is_terminated()
        self._is_truncated()

        return self.observation, self.reward, self._terminated, self._truncated, {}

    def render(self, mode='human'):
        """
        Render the environment
        """
        if mode == 'human':
            self.vis.BeginScene()
            self.vis.Render()
            self.vis.EndScene()
        else:
            raise NotImplementedError

    def get_reward(self):
        """
        Get the reward for the current step
        """
        scale = 1
        # Distance to goal
        distance = np.linalg.norm(self.observation - self.goal)
        # If we are closer to the goal than before -> Positive reward based on how much closer
        # If further away -> Negative reward based on how much further
        reward = scale * (self._old_distance - distance)

        # Update the old distance
        self._old_distance = distance

        return reward

    def _is_terminated(self):
        """
        Check if the environment is terminated
        """
        # If we have exceeded the max time -> Terminate
        if self.system.GetChTime() > self._max_time:
            print('Time out')
            # Penalize based on how far we are from the goal
            self.reward -= 100 * np.linalg.norm(self.observation - self.goal)
            self._terminated = True

        # If we are within a certain distance of the goal -> Terminate and give big reward
        if np.linalg.norm(self.observation - self.goal) < 1:
            print('Goal Reached')
            self.reward += 1000
            self._terminated = True

    def _is_truncated(self):
        """
        Check if the environment is truncated
        """
        # If we have collided -> Truncate and give big negative reward
        self.check_collision()
        if self._collision:
            print('Crashed')
            self.reward -= 1000
            self._truncated = True

    def initialize_robot_pos(self, seed=1):
        """
        Initialize the pose of the robot
        """
        # For now no randomness
        self.rover.Initialize(chrono.ChFrameD(chrono.ChVectorD(
            0, -0.2, -0.3), chrono.ChQuaternionD(1, 0, 0, 0)))

    def set_goalPoint(self, seed=1):
        """
        Set the goal point for the rover
        """
        # Some random goal point for now
        self.goal = np.array([1, 1, 0.5])

    def get_observation(self):
        """
        Get the observation from the environment
        """
        # For not just the priveledged position of the rover
        return chVector_to_npArray(self.rover.GetChassis().GetPos())

    # ------------------------------------- TODO: Add Random Objects to the environment -------------------------------------

    def add_obstacles(self, seed=1):
        """
        Add random obstacles to the environment
        """
        pass

    # ------------------------------------- TODO: Add Sensors if necessary -------------------------------------
    def add_sensors(self):
        """
        Add sensors to the rover
        """

        pass

    # ------------------------------------- TODO: Check for collision with objects -------------------------------------
    def check_collision(self):
        """
        Check if we collide with any of the objects
        """
        self._collision = False
