# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

with open("policy_q.pkl", "rb") as file:
    q_table = pickle.load(file)
steps = 0
pickup = False

def get_state(obs, pickup):
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
    state += [
        pickup
    ]
    return tuple(state)

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    exp_x = np.exp(x - np.max(x))  # Subtract max(x) for numerical stability
    return exp_x / np.sum(exp_x)

def get_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    steps += 1
    state = get_state(obs, pickup)
    if state not in q_table:
        action = np.random.choice(6) # Choose a random action
    else:
        #p = softmax(q_table[state])
        #action = np.random.choice(6, p=p)        
        if np.random.rand() < 0.005:
            action = np.random.choice(6)
        else:
            action = np.argmax(q_table[state])
    if not pickup:
        at_pickup = state[12] and ((state[0] == 0 and state[1] == 0) or (state[2] == 0 and state[3] == 0) \
            or (state[4] == 0 and state[5] == 0) or (state[6] == 0 and state[7] == 0))
        action_pickup = action == 4
        pickup = at_pickup and action_pickup
    else:
        at_destination = state[13] and ((state[0] == 0 and state[1] == 0) or (state[2] == 0 and state[3] == 0) \
            or (state[4] == 0 and state[5] == 0) or (state[6] == 0 and state[7] == 0))
        action_drop = action == 5
        pickup = not(at_destination and action_drop)
    if steps >= 5000:
        pickup = False
        steps = 0
    return action
    # You can submit this random agent to evaluate the performance of a purely random strategy.
