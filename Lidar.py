#! /usr/bin/env python

import numpy as np
from math import *
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan

MAX_LIDAR_DISTANCE = 1.0
COLLISION_DISTANCE = 0.14 # LaserScan.range_min = 0.1199999
NEARBY_DISTANCE = 0.45

ZONE_0_LENGTH = 0.4
ZONE_1_LENGTH = 0.7

ANGLE_MAX = 360 - 1
ANGLE_MIN = 1 - 1
HORIZON_WIDTH = 75

Max_distance = 5

# Convert LasecScan msg to array
def lidarScan(msgScan):
    distances = np.array([])
    angles = np.array([])

    for i in range(len(msgScan.ranges)):
        angle = degrees(i * msgScan.angle_increment)
        if ( msgScan.ranges[i] > MAX_LIDAR_DISTANCE ):
            distance = MAX_LIDAR_DISTANCE
        elif ( msgScan.ranges[i] < msgScan.range_min ):
            distance = msgScan.range_min
            # For real robot - protection
            if msgScan.ranges[i] < 0.01:
                distance = MAX_LIDAR_DISTANCE
        else:
            distance = msgScan.ranges[i]

        distances = np.append(distances, distance)
        angles = np.append(angles, angle)

    # distances in [m], angles in [degrees]
    return ( distances, angles )

# Discretization of lidar scan
def scanDiscretization(state_space, lidar):
    x1 = 2 # Left zone (no obstacle detected)
    x2 = 2 # Right zone (no obstacle detected)
    x3 = 3 # Left sector (no obstacle detected)
    x4 = 3 # Right sector (no obstacle detected)

    # Find the left side lidar values of the vehicle
    lidar_left = min(lidar[(ANGLE_MIN):(ANGLE_MIN + HORIZON_WIDTH)])
    if ZONE_1_LENGTH > lidar_left > ZONE_0_LENGTH:
        x1 = 1 # zone 1
    elif lidar_left <= ZONE_0_LENGTH:
        x1 = 0 # zone 0

    # Find the right side lidar values of the vehicle
    lidar_right = min(lidar[(ANGLE_MAX - HORIZON_WIDTH):(ANGLE_MAX)])
    if ZONE_1_LENGTH > lidar_right > ZONE_0_LENGTH:
        x2 = 1 # zone 1
    elif lidar_right <= ZONE_0_LENGTH:
        x2 = 0 # zone 0

    # Detection of object in front of the robot
    if ( min(lidar[(ANGLE_MAX - HORIZON_WIDTH // 3):(ANGLE_MAX)]) < 1.0 ) or ( min(lidar[(ANGLE_MIN):(ANGLE_MIN + HORIZON_WIDTH // 3)]) < 1.0 ):
        object_front = True
    else:
        object_front = False

    # Detection of object on the left side of the robot ?
    if min(lidar[(ANGLE_MIN):(ANGLE_MIN + 2 * HORIZON_WIDTH // 3)]) < 1.0:
        object_left = True
    else:
        object_left = False

    # Detection of object on the right side of the robot ?
    if min(lidar[(ANGLE_MAX - 2 * HORIZON_WIDTH // 3):(ANGLE_MAX)]) < 1.0:
        object_right = True
    else:
        object_right = False

    # Detection of object on the far left side of the robot
    if min(lidar[(ANGLE_MIN + HORIZON_WIDTH // 3):(ANGLE_MIN + HORIZON_WIDTH)]) < 1.0:
        object_far_left = True
    else:
        object_far_left = False

    # Detection of object on the far right side of the robot
    if min(lidar[(ANGLE_MAX - HORIZON_WIDTH):(ANGLE_MAX - HORIZON_WIDTH // 3)]) < 1.0:
        object_far_right = True
    else:
        object_far_right = False

    # The left sector of the vehicle
    if ( object_front and object_left ) and ( not object_far_left ):
        x3 = 0 # sector 0
    elif ( object_left and object_far_left ) and ( not object_front ):
        x3 = 1 # sector 1
    elif object_front and object_left and object_far_left:
        x3 = 2 # sector 2

    if ( object_front and object_right ) and ( not object_far_right ):
        x4 = 0 # sector 0
    elif ( object_right and object_far_right ) and ( not object_front ):
        x4 = 1 # sector 1
    elif object_front and object_right and object_far_right:
        x4 = 2 # sector 2

    # Find the state space index of (x1,x2,x3,x4) in Q table
    ss = np.where(np.all(state_space == np.array([x1,x2,x3,x4]), axis = 1))
    state_ind = int(ss[0])

    return ( state_ind, x1, x2, x3 , x4 )

# Check - crash
def checkCrash(lidar):
    lidar_horizon = np.concatenate((lidar[(ANGLE_MIN + HORIZON_WIDTH):(ANGLE_MIN):-1],lidar[(ANGLE_MAX):(ANGLE_MAX - HORIZON_WIDTH):-1]))
    W = np.linspace(1.2, 1, len(lidar_horizon) // 2)
    W = np.append(W, np.linspace(1, 1.2, len(lidar_horizon) // 2))
    if np.min( W * lidar_horizon ) < COLLISION_DISTANCE:
        return True
    else:
        return False

# Check - object nearby
def checkObjectNearby(lidar):
    lidar_horizon = np.concatenate((lidar[(ANGLE_MIN + HORIZON_WIDTH):(ANGLE_MIN):-1],lidar[(ANGLE_MAX):(ANGLE_MAX - HORIZON_WIDTH):-1]))
    W = np.linspace(1.4, 1, len(lidar_horizon) // 2)
    W = np.append(W, np.linspace(1, 1.4, len(lidar_horizon) // 2))
    if np.min( W * lidar_horizon ) < NEARBY_DISTANCE:
        return True
    else:
        return False

# Check - goal near
def checkGoalNear(x, y, x_goal, y_goal):
    ro = sqrt( pow( ( x_goal - x ) , 2 ) + pow( ( y_goal - y ) , 2) )
    if ro < 0.3:
        return True
    else:
        return False

def getScan_16(self): 

        """ Ladar by P'Almon this function returns list of 16 floats representing distance
            at each angle from 0 to pi/8, 2pi/8, 3pi/8, ... to 16pi/8
        """
        
        _, msg = self.wait_for_message(topic='/scan', msg_type=LaserScan)
        incre = msg.angle_increment
        # print("incre =",incre)

        # get distances of each angle of interest
        distances = []
        for i in range(16):

            MAX_LIDAR_DISTANCE = 12.0

            if i!= 0:
                angle = i*pi/8
                i = int(angle/incre) # just 16 directions of interest

            # print("msg.ranges[",i,"] =",msg.ranges[i], type(msg.ranges[i])) ### CHECK INF

            distance = msg.ranges[i]

            if (msg.ranges[i] > MAX_LIDAR_DISTANCE):
                # print("greater than max")
                distance = MAX_LIDAR_DISTANCE
                # print("distance =",distance)
            distances.append(distance)

        # distances in [m], angles in [degrees]
        # print(distances)
        return distances
    

def getlidar10(msg):
    # incre = msg.angle_increment * 15
    distances = np.array([])
    
    for j in range(2):
        for i in range(5):
            idx = i * 15
            if(j):
                close_dist = min(msg.ranges[ANGLE_MAX - idx - 15: ANGLE_MAX - idx])
            else:
                close_dist = min(msg.ranges[idx: idx + 15])
            # print(close_dist)
            if(close_dist > Max_distance):
                close_dist = Max_distance
            distances = np.append(distances, close_dist / Max_distance)

    return distances

def convert451to360(msg):
    distances = np.array([])

    for i in range(len(msg.ranges)):
        if ( msg.ranges[i] > Max_distance):
            distance = Max_distance
        elif ( msg.ranges[i] < msg.range_min):
            distance = msg.range_min
            # For real robot - protection
        else:
            distance = msg.ranges[i]

        distances = np.append(distances, distance)
        
    # Create an array of indices for the original list
    indices = np.arange(len(distances))

    # Create an array of indices for the new list
    new_indices = np.linspace(0, len(distances) - 1, 360)

    # Use linear interpolation to compute the values for the new list
    new_list = np.interp(new_indices, indices, distances)
    
    for i in range(len(new_list)):
        if isnan(new_list[i]):
            new_list[i] = Max_distance
            
    return new_list

def convert360to10(msg):
    new_distances = np.array([])
    for j in range(2):
        for i in range(5):
            idx = i * 15
            if(j):
                close_dist = min(msg[ANGLE_MAX - idx - 15: ANGLE_MAX - idx])
            else:
                close_dist = min(msg[idx: idx + 15])
            # print(close_dist)
            if(close_dist > Max_distance):
                close_dist = Max_distance
                
            new_distances = np.append(new_distances, close_dist)
            
    return new_distances
    