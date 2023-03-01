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
from Spawngoal import *

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

from math import dist, pi


X_INIT = 0.0
Y_INIT = 0.0
THETA_INIT = 0.0

X_GOAL = 3.0
Y_GOAL = 3.0
GETGOALTHRESHOLD = 0.7

ANGLE_MAX = 359
ANGLE_MIN = 0
HORIZON_WIDTH = 75
DIST_MAX = 6
ANGLE_YAW_NORM = 2 * pi


class Publisher():
    def __init__(self, node: Node):
        self.velPub = node.create_publisher(Twist, 'cmd_vel', 10)
        self.setPosPub = node.create_publisher(ModelState, 'gazebo/set_model_state', 10)
        self.resetWorld = node.create_client(Empty, '/reset_simulation')
        
class rosworld(Env):
    def __init__(self):
        self.node = Node("rosworld")
        
    
        self.publisher = Publisher(self.node)
        self.action_space = spaces.Box(low =np.array([0.08, -0.5]), high = np.array([0.2, 0.5]), shape=(2, ) , dtype = np.float32)
        self.observation_space = spaces.Box(low = np.NINF,
            high = np.Inf, shape=(12, ), dtype=np.float32)
        
        self.step_cnt = 0
        self.crash_count = 0
        self.getgoal_count = 0
        self.timeout_count = 0
        
        self.x_goal = X_GOAL
        self.y_goal = Y_GOAL
        self.prev_lidar = np.zeros(360)
        self.prev_action = 0
        self.step_per_reset = 0
        self.done = False
        
    def step(self, action):
        d1 = self.observation[0]
        velMsg = createVelMsg(float(action[0]), float(action[1]))
        self.publisher.velPub.publish(velMsg)
        
        _, odomMsg = self.wait_for_message('/odom', Odometry)
        _, msgScan = self.wait_for_message('/scan', LaserScan)
        lidar= convert451to360(msgScan)
        lidar2 = convert360to10(lidar)
        ( self.x, self.y ) = getPosition(odomMsg)
        self.yaw = getRotation(odomMsg)
        
        self.dist = dist([self.x, self.y], [self.x_goal, self.y_goal])
        self.observation = np.array([self.dist/ DIST_MAX, self.yaw / ANGLE_YAW_NORM] + lidar2.tolist())
        
        info = {}
        robot_coor = [self.x, self.y]
        goal_coor = [self.x_goal, self.y_goal]
        self.reward = 0
        
        if checkCrash(lidar):
            # robotStop(self.publisher.velPub)
            print("CrashHHHHHHHH!!!!")
            self.crash_count += 1
            # self.done = True
            velMsg = createVelMsg(-float(action[0] * 1.5), 0.0)
            self.publisher.velPub.publish(velMsg)   
            sleep(1)
            
        
        elif getGoal(GETGOALTHRESHOLD, robot_coor, goal_coor):
            robotStop(self.publisher.velPub)
            print("GOALLLLLLLLLL!!!!")
            self.getgoal_count += 1
            self.done = True
            

        if self.step_per_reset > 999:
            print('TIMEOUTTTTTT!!!!!')
            # self.reward += 100
            # self.done = False
        
        self.prev_lidar = lidar
        self.prev_action = action[1]
        
        self.step_cnt += 1
        self.step_per_reset += 1
        #check
        print(f'step_count: {self.step_cnt}')
        print(f'step_per_reset: {self.step_per_reset}')
        print(f'action: {action[0]:.2f} {action[1]:.2f}')
        print(f"X: {self.x:.2f} => {self.x_goal:.2f}")
        print(f"Y: {self.y:.2f} => {self.y_goal:.2f}")
        print(f"dist: {self.observation[0]:.2f}")
        print(f"crash_count: {self.crash_count}")
        print(f"getgoal_count: {self.getgoal_count}")
        print("###########################################")
        return self.observation, self.reward, self.done, info

    def reset(self):
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
        
        self.step_per_reset = 0
        self.done = False
        self.prev_lidar = np.zeros(360)
        self.prev_action = 0
        _, odomMsg = self.wait_for_message('/odom', Odometry)
        _, msgScan = self.wait_for_message('/scan', LaserScan)
        
        lidar = convert360to10(convert451to360(msgScan))
        
        # reset goal
        self.x_goal = X_GOAL
        self.y_goal = Y_GOAL
        delete_circle()
        sleep(0.2)
        spawn_circle([self.x_goal, self.y_goal])
        # get init dist from robot to goal
        self.dist = dist([self.x, self.y],[self.x_goal, self.y_goal])
        self.yaw = getRotation(odomMsg)
        
        self.observation = np.array([self.dist/ DIST_MAX, self.yaw / ANGLE_YAW_NORM] + lidar.tolist())
        # self.observation = np.array(lidar)
        
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

env = rosworld()
env.reset()

model_path = '/home/machi01/models/PPOv1/30000.zip'
model = PPO.load(model_path, env=env)


# obs = env.reset()
# done = False
obs = env.reset()
done = False
while not done:
    action, _state = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    # print(rewards)
#v100 random 3 goal position with 10lidar(150 degrees) all