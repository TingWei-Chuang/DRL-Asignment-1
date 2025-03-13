# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

with open("testq_stable.pkl", "rb") as file:
    q_table = pickle.load(file)

def get_state(obs):
    taxi_row, taxi_col, station_0_row, station_0_col, station_1_row, station_1_col, station_2_row, station_2_col, station_3_row, station_3_col, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look  = \
        obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], obs[6], obs[7], obs[8], obs[9], obs[10], obs[11], obs[12], obs[13], obs[14], obs[15]
    state = []
    def dir(a, b):
        if a < b:
            return -1
        elif a > b:
            return 1
        return 0
    def nearby(x1, y1, x2, y2):
        if x1 == x2 - 1 and y1 == y2:
            return 1
        elif x1 == x2 + 1 and y1 == y2:
            return 2
        elif x1 == x2 and y1 == y2 - 1:
            return 3
        elif x1 == x2 and y1 == y2 + 1:
            return 4
        else:
            return 0
    state += [
        dir(taxi_row, station_0_row),
        dir(taxi_col, station_0_col),
        dir(taxi_row, station_1_row),
        dir(taxi_col, station_1_col),
        dir(taxi_row, station_2_row),
        dir(taxi_col, station_2_col),
        dir(taxi_row, station_3_row),
        dir(taxi_col, station_3_col)
    ]
    state += [
        obstacle_north,
        obstacle_south,
        obstacle_east,
        obstacle_west
    ]
    state += [
        passenger_look,
        destination_look
    ]
    state += [
        nearby(taxi_row, taxi_col, station_0_row, station_0_col),
        nearby(taxi_row, taxi_col, station_1_row, station_1_col),
        nearby(taxi_row, taxi_col, station_2_row, station_2_col),
        nearby(taxi_row, taxi_col, station_3_row, station_3_col)
    ]
    return tuple(state)

def get_action(obs):
    state = get_state(obs, pickup, visitsA, visitsB)
    if state not in q_table:
        action = np.random.choice(6) # Choose a random action
    else:
        if np.random.rand() < 0.0:
            action = np.random.choice(6)
        else:
            action = np.argmax(q_table[state])
    return action
    # You can submit this random agent to evaluate the performance of a purely random strategy.
