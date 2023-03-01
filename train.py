import gym 
from gym import Env
from gym import spaces
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete 
import numpy as np
from numpy import dot
from numpy.linalg import norm
import random
import os
from time import sleep
from stable_baselines3 import PPO
# from sb3_contrib import ARS, QRDQN, TQC, TRPO, MaskablePPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from Control import *
from Lidar import *
from Spawngoaltrain import *

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose, Point, Quaternion
from nav_msgs.msg import Odometry
from gazebo_msgs.msg._model_state import ModelState

from std_srvs.srv import Empty
from std_srvs.srv._empty import Empty_Request

from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerGuardCondition
from rclpy.utilities import timeout_sec_to_nsec

from geometry_msgs.msg import Pose, Point, Quaternion
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from math import dist, pi


#init const value
X_INIT = 0.0
Y_INIT = 0.0
THETA_INIT = 0.0

# X_GOAL = 1.5
# Y_GOAL = -1.5
X_GOAL_RAND = [0.0, 3.5, 3.5]
Y_GOAL_RAND = [3.5, 3.5, 0.0]
GETGOALTHRESHOLD = 0.4

ANGLE_MAX = 359
ANGLE_MIN = 0
HORIZON_WIDTH = 75
DIST_MAX = 6
ANGLE_YAW_NORM = 2 * pi

# %%
class Publisher():
    def __init__(self, node: Node):
        self.velPub = node.create_publisher(Twist, 'cmd_vel', 10)
        self.setPosPub = node.create_publisher(ModelState, 'gazebo/set_model_state', 10)
        self.resetWorld = node.create_client(Empty, '/reset_simulation')
        
class rosworld(Env):
    def __init__(self):
        self.node = Node("rosworld")
        #call Superclass
        self.publisher = Publisher(self.node)
        self.action_space = spaces.Box(low =np.array([0.08, -0.5]), high = np.array([0.2, 0.5]), shape=(2, ) , dtype = np.float32)
        self.observation_space = spaces.Box(low = np.NINF,
            high = np.Inf, shape=(12, ), dtype=np.float32)
        
        self.step_cnt = 0
        self.crash_count = 0
        self.getgoal_count = 0
        self.timeout_count = 0
        # self.x = 0
        # self.y = 0
        rand_num = random.randint(0, 2)
        self.x_goal = X_GOAL_RAND[rand_num]
        self.y_goal = Y_GOAL_RAND[rand_num]
        self.prev_lidar = np.zeros(360)
        self.prev_action = 0
        self.step_per_reset = 0
        self.done = False
        
        

    def step(self, action):
        d1 = self.observation[0]
        #Do action velocity
        velMsg = createVelMsg(float(action[0]), float(action[1]))
        self.publisher.velPub.publish(velMsg)
        
        _, odomMsg = self.wait_for_message('/odom', Odometry)
        _, msgScan = self.wait_for_message('/scan', LaserScan)
        ( lidar, angles ) = lidarScan(msgScan)
        lidar2 = getlidar10(msgScan)
        ( self.x, self.y ) = getPosition(odomMsg)
        self.yaw = getRotation(odomMsg)
        
        self.dist = dist([self.x, self.y], [self.x_goal, self.y_goal])
        self.observation = np.array([self.dist / DIST_MAX, self.yaw / ANGLE_YAW_NORM]+lidar2.tolist())
        
        info = {}
        #####reward and condition
        robot_coor = [self.x, self.y]
        goal_coor = [self.x_goal, self.y_goal]
        #coSine is cos value between goal and robot
        coSine = dot(robot_coor, goal_coor)/(norm(robot_coor)*norm(goal_coor))
        self.reward = (coSine * 2 + (self.dist - d1))/12
        # self.reward = 0
                # Reward from action taken = fowrad -> +0.2 , turn -> -0.1
        if abs(action[1]) <= 0.1:
            self.reward += 0.1
        else:
            self.reward -= 0.05
            
        # Reward from crash distance to obstacle change
        #lidar reward
        lidar_horizon = np.concatenate((lidar[(ANGLE_MIN + HORIZON_WIDTH):(ANGLE_MIN):-1],lidar[(ANGLE_MAX):(ANGLE_MAX - HORIZON_WIDTH):-1]))
        prev_lidar_horizon = np.concatenate((self.prev_lidar[(ANGLE_MIN + HORIZON_WIDTH):(ANGLE_MIN):-1],self.prev_lidar[(ANGLE_MAX):(ANGLE_MAX - HORIZON_WIDTH):-1]))

        W = np.linspace(0.9, 1.1, len(lidar_horizon) // 2)
        W = np.append(W, np.linspace(1.1, 0.9, len(lidar_horizon) // 2))
        if np.sum( W * ( lidar_horizon - prev_lidar_horizon) ) >= 0:
            self.reward += 0.2
        else:
            self.reward -= 0.2
            
        # Reward from turn left/right change
        if abs(action[1] - self.prev_action) >= 0.8:
            self.reward -= 0.8
        

        if checkCrash(lidar):
            robotStop(self.publisher.velPub)
            print("CrashHHHHHHHH!!!!")
            self.crash_count += 1
            self.reward -= 100
            self.done = True
        
        elif getGoal(GETGOALTHRESHOLD, robot_coor, goal_coor):
            print("GOALLLLLLLLLL!!!!")
            self.getgoal_count += 1
            self.reward += 1000
            self.done = True

        if self.step_per_reset > 499:
            print('TIMEOUTTTTTT!!!!!')
            self.reward += 100
            self.timeout_count += 1
            self.done = True
        
        #save lidar and action to estimate reward
        self.prev_lidar = lidar
        self.prev_action = action[1]

        self.step_cnt += 1
        self.step_per_reset += 1
        #debugging
        print(f'step_count: {self.step_cnt}')
        print(f'step_per_reset: {self.step_per_reset}')
        print(f'action: {action[0]:.2f} {action[1]:.2f}')
        print(f"X: {self.x:.2f} => {self.x_goal:.2f}")
        print(f"Y: {self.y:.2f} => {self.y_goal:.2f}")
        print(f"dist: {self.observation[0]:.2f}")
        print(f"cosine: {coSine:.2}")
        print(f"reward: {self.reward:.2f}")
        print(f"crash_count: {self.crash_count}")
        print(f"getgoal_count: {self.getgoal_count}")
        print(f"timeout_count: {self.timeout_count}")
        print("###########################################")
        return self.observation, self.reward, self.done, info
        
    
    def reset(self):
        #pidnode, then init
        pidnode(self.node)
        
        rclpy.init()
        self.node = Node("rosworld")
        self.publisher = Publisher(self.node)
        
        while True:
            self.publisher.resetWorld.call_async(Empty_Request())
            robotSetPos(self.publisher.setPosPub, X_INIT, Y_INIT, THETA_INIT)
            # get init position -> [x, y]
            _, odomMsg = self.wait_for_message('/odom', Odometry)
            _, msgScan = self.wait_for_message('/scan', LaserScan)
            (self.x, self.y) = getPosition(odomMsg)
            print(f'reset {self.x, self.y}')
            sleep(0.5)
            if (abs(self.x - X_INIT) <= 0.5) & (abs(self.y - Y_INIT) <= 0.5):
                break
        
        #delete circle
        delete_circle()
        
        self.step_per_reset = 0
        self.done = False
        self.prev_lidar = np.zeros(360)
        self.prev_action = 0
        _, odomMsg = self.wait_for_message('/odom', Odometry)
        _, msgScan = self.wait_for_message('/scan', LaserScan)
        lidar = getlidar10(msgScan)
        
        # reset goal
        # self.x_goal = random.uniform(-2., 2.)
        # self.y_goal = random.uniform(-2., 2.)
        rand_num = random.randint(0, 2)
        self.x_goal = X_GOAL_RAND[rand_num]
        self.y_goal = Y_GOAL_RAND[rand_num]
        
        #create circle
        sleep(0.2)
        spawn_circle([self.x_goal, self.y_goal])
        
        
        # get init dist from robot to goal
        self.dist = dist([self.x, self.y],[self.x_goal, self.y_goal])
        self.yaw = getRotation(odomMsg)
        
        self.observation = np.array([self.dist/ DIST_MAX, self.yaw / ANGLE_YAW_NORM] + lidar.tolist())

        return self.observation


    def wait_for_message(
        self,
        topic: str,
        msg_type,
        time_to_wait=-1
    ):
        """
        Wait for the next incoming message.
        :param msg_type: message type
        :param node: node to initialize the subscription on
        :param topic: topic name to wait for message
        :time_to_wait: seconds to wait before returning
        :return (True, msg) if a message was successfully received, (False, ()) if message
            could not be obtained or shutdown was triggered asynchronously on the context.
        """
        context = self.node.context
        wait_set = _rclpy.WaitSet(1, 1, 0, 0, 0, 0, context.handle)
        wait_set.clear_entities()

        sub = self.node.create_subscription(msg_type, topic, lambda _: None, 1)
        wait_set.add_subscription(sub.handle)
        sigint_gc = SignalHandlerGuardCondition(context=context)
        wait_set.add_guard_condition(sigint_gc.handle)

        timeout_nsec = timeout_sec_to_nsec(time_to_wait)
        wait_set.wait(timeout_nsec)

        subs_ready = wait_set.get_ready_entities('subscription')
        guards_ready = wait_set.get_ready_entities('guard_condition')

        if guards_ready:
            if sigint_gc.handle.pointer in guards_ready:
                return (False, None)

        if subs_ready:
            if sub.handle.pointer in subs_ready:
                msg_info = sub.handle.take_message(sub.msg_type, sub.raw)
                return (True, msg_info[0])

        return (False, None)
         
def pidnode(node):
    node.destroy_node()
    rclpy.shutdown()

def getGoal(get_goal_threshold, robot_coor, goal_coor):
    output = abs(dist(robot_coor,goal_coor))
    if output <= get_goal_threshold:
        return True
    else:
        return False

# %%
#init ros2
if not rclpy.ok():
    rclpy.init()

#just change version to save new paths
version = 3
models_dir = f"models/PPOv{version}"
logdir = "logs"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = rosworld()
env.reset()

# ent_coef greater, explore more

# model = PPO('MlpPolicy', env, verbose=1, n_steps = 200, tensorboard_log=logdir, ent_coef = 0.1)
model = PPO.load("/home/machi01/3_0.zip", env)

TIMESTEPS = 10000
for i in range(10):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPOv{version}")
    model.save(f"{models_dir}/{TIMESTEPS*i}")

#v100 random 3 goal position with 10lidar(150 degrees) all