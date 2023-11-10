# =======================================================================================
# PROJECT CHRONO - http://projectchrono.org
#
# Copyright (c) 2021 projectchrono.org
# All right reserved.
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE file at the top level of the distribution and at
# http://projectchrono.org/license-chrono.txt.
#
# =======================================================================================
# Authors: Huzaifa Unjhawala, Jason Zhou
# =======================================================================================
#
# This file contains a gym environment for the cobra rover in a terrain of 20 x 20. The
# environment is used to train the rover to reach a goal point in the terrain. The goal
# point is randomly generated in the terrain. The rover is initialized at the center of
# the terrain. Obstacles can be optionally set (default is 0).
#
# =======================================================================================
#
# Action Space: The action space is normalized throttle and steering between -1 and 1.
# multiply against the max wheel angular velocity and wheel steer angle to provide the
# wheel angular velocity and wheel steer angle for all 4 wheels of the cobra rover model.
# Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64)
#
# =======================================================================================
#
# Observation Space: The observation space is a 1D array consisting of the following:
# 1. Delta x of the goal in local frame of the vehicle
# =======================================================================================


# Chrono imports
from gymnasium.core import Env
import pychrono as chrono
import pychrono.robot as robot_chrono
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


class cobra_wpts(ChronoBaseEnv):
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

        # Define action space -> Now only steering
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float64)

        # Define observation space
        # For now only the error to the way point
        self.observation_space = gym.spaces.Box(
            low=-20, high=20, shape=(1,), dtype=np.float64)

        # -----------------------------
        # Chrono simulation parameters
        # -----------------------------
        self.system = None  # Chrono system set in reset method
        self.ground = None  # Ground body set in reset method
        self.rover = None  # Rover set in reset method

        self._initpos = chrono.ChVectorD(
            0.0, 0.0, 0.3)  # Rover initial position
        # Frequncy in which we apply control
        self._control_frequency = 5
        # Dynamics timestep
        self._step_size = 5e-4
        # Number of steps dynamics has to take before we apply control
        self._steps_per_control = round(
            1 / (self._step_size * self._control_frequency))
        self._collision = False
        self._terrain_length = 60
        self._terrain_width = 60
        self._terrain_height = 0.4
        self.rover_pos = None

        # ---------------------------------
        # Gym Environment variables
        # ---------------------------------
        # Maximum simulation time (seconds)
        self._max_time = 180
        # Holds reward of the episode
        self.reward = 0
        self._debug_reward = 0

        # Observation of the environment
        self.observation = None
        # Flag to determine if the environment has terminated -> In the event of timeOut or reach goal
        self._terminated = False
        # Flag to determine if the environment has truncated -> In the event of a crash
        self._truncated = False
        # Flag to check if the render setup has been done -> Some problem if rendering is setup in reset
        self._render_setup = False
        
        # Case specific
        self.lookahead = 3.0
        self.path_file_name = "/home/jason/Desktop/STUDY/main_fork/chrono/data/robot/environment/room_1/PATH.txt"
        
        #Waypoints
        self.x_coords, self.y_coords, self.z_coords = [], [], []
    
        

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state -> Set up for standard gym API
        :param seed: Seed for the random number generator
        :param options: Options for the simulation (dictionary)
        """
        
        # Read waypoints from file
        def read_waypoints(file_path):
            x_coords, y_coords, z_coords = [], [], []
            try:
                with open(file_path, 'r') as file:
                    for line in file:
                        x, y, z = line.strip().split(' ')
                        x_coords.append(float(x))
                        y_coords.append(float(y))
                        z_coords.append(float(z))
                return x_coords, y_coords, z_coords
            except FileNotFoundError:
                self.get_logger().error(f"File not found: {file_path}")
                return [], [], []
            except Exception as e:
                self.get_logger().error(f"An error occurred: {e}")
                return [], [], []
            
        self.x_coords, self.y_coords, self.z_coords = read_waypoints(self.path_file_name)



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

        # -----------------------------
        # Create the COBRA
        # -----------------------------
        self.rover = robot_chrono.Cobra(
            self.system, robot_chrono.CobraWheelType_SimpleWheel)
        self.driver = robot_chrono.CobraSpeedDriver(
            1/self._control_frequency, chrono.CH_C_PI/4)
        self.rover.SetDriver(self.driver)

        # Initialize position of robot
        self._initpos = chrono.ChVectorD(-4.165707, -17.150379, 0.4)

        # For now no randomness
        self.rover.Initialize(chrono.ChFrameD(
            self._initpos, chrono.ChQuaternionD(1, 0, 0, 0)))

        self.cur_pos = self._initpos
        
        # -----------------------------
        # Visualize way points
        # -----------------------------
        # -----------------------------
        # Set up goal visualization
        # -----------------------------
        for i in range(len(self.x_coords)):
            goal_contact_material = chrono.ChMaterialSurfaceNSC()
            goal_mat = chrono.ChVisualMaterial()
            goal_mat.SetAmbientColor(chrono.ChColor(1., 0., 0.))
            goal_mat.SetDiffuseColor(chrono.ChColor(1., 0., 0.))

            goal_body = chrono.ChBodyEasySphere(
                0.1, 1000, True, False, goal_contact_material)

            goal_body.SetPos(chrono.ChVectorD(
                self.x_coords[i], self.y_coords[i], 0.2))
            goal_body.SetBodyFixed(True)
            goal_body.GetVisualShape(0).SetMaterial(0, goal_mat)

            self.system.Add(goal_body)

        # -----------------------------
        # Get the intial observation
        # -----------------------------
        self.update_headingerror()
        self.observation = self.get_observation()

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
        self.driver.SetSteering(steer_angle)  # Maybe we should ramp this steer

        for i in range(self._steps_per_control):
            self.rover.Update()
            self.system.DoStepDynamics(self._step_size)
            self.update_headingerror()
            


        # Get the observation
        self.observation = self.get_observation()
        # Get reward
        self.reward = self.get_reward()
        self._debug_reward += self.reward
        # Check if we are done
        self._is_terminated()
        self._is_truncated()

        return self.observation, self.reward, self._terminated, self._truncated, {}

    def render(self, mode='human'):
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
                    0, 11, 10), chrono.ChVectorD(0, 0, 1))
                self.vis.AddTypicalLights()
                self.vis.AddLightWithShadow(chrono.ChVectorD(
                    1.5, -2.5, 5.5), chrono.ChVectorD(0, 0, 0.5), 3, 4, 10, 40, 512)
                self._render_setup = True

            self.vis.BeginScene()
            self.vis.Render()
            self.vis.EndScene()
        else:
            raise NotImplementedError

    def get_reward(self):
        """
        Get the reward for the current step
        """
        
        max_heading_err = np.pi
        
        reward = 1 - (abs(self.heading_error)/max_heading_err)
        
        return reward

    def _is_terminated(self):
        """
        Check if the environment is terminated
        """
        # If we have exceeded the max time -> Terminate
        if self.system.GetChTime() > self._max_time:
            print('--------------------------------------------------------------')
            print('Time out')
            print('Accumulated Reward: ', self._debug_reward)
            print('--------------------------------------------------------------')
            self._terminated = True
        else:
            self._terminated = False

    def _is_truncated(self):
        """
        Check if the environment is truncated
        """
        self._truncated = False

    def get_observation(self):
        """
        Get the observation from the environment
            1. Steer Error to the way point
        """
        observation = np.zeros(1)


        observation[0] = self.heading_error
     

        # For not just the priveledged position of the rover
        return observation

    def update_headingerror(self):
        """
        Update the heading error
        """
        self.rover_pos = self.rover.GetChassis().GetPos()
        self.rover_rot = self.rover.GetChassis().GetRot()
        self.rover_rot_euler = chrono.Q_to_Euler123(self.rover_rot)
            
        yaw = self.rover_rot_euler.z
           
        forward_x = math.cos(yaw)
        forward_y = math.sin(yaw)

        x_ahead = self.rover_pos.x + self.lookahead  * forward_x
        y_ahead = self.rover_pos.y + self.lookahead *  forward_y
            
        min_distance = float('inf')
        closest_waypoint = None
            
        def euclidean_distance(x1, y1, x2, y2):
            return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        for x, y, z in zip(self.x_coords, self.y_coords, self.z_coords):
            distance = euclidean_distance(x, y, x_ahead, y_ahead)
            if distance < min_distance:
                min_distance = distance
                closest_waypoint = (x, y, z)
            
        desired_heading = math.atan2(
            closest_waypoint[1] - y_ahead,
            closest_waypoint[0] - x_ahead
        )
            
        # Calculate heading error
        self.heading_error = desired_heading - yaw

        # Normalize the error to the range [-pi, pi]
        self.heading_error = (self.heading_error + math.pi) % (2 * math.pi) - math.pi
        