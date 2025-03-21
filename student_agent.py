# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

with open("qqqqqqqqq.pkl", "rb") as file:
    q_table = pickle.load(file)
steps = 0
pickup = False
visitsA = 0
visitsB = 0
prevAction = 0

def get_state(obs, pickup, visitsA, visitsB, prevAction):
    #, station_3_row, station_3_col
    taxi_row, taxi_col, station_0_row, station_0_col, station_1_row, station_1_col, station_2_row, station_2_col, station_3_row, station_3_col, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look  = \
        obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], obs[6], obs[7], obs[8], obs[9], obs[10], obs[11], obs[12], obs[13], obs[14], obs[15]
    stations = [(station_0_row, station_0_col), (station_1_row, station_1_col), (station_2_row, station_2_col), (station_3_row, station_3_col)]
    if not pickup:
        station_row = stations[visitsA][0]
        station_col = stations[visitsA][1]
    else:
        station_row = stations[visitsB][0]
        station_col = stations[visitsB][1]
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
        dir(taxi_row, station_row),
        dir(taxi_col, station_col),
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
        nearby(taxi_row, taxi_col, station_row, station_col)
    ]
    state += [
        pickup
    ]
    state += [
        prevAction
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
    global steps
    global pickup
    global visitsA
    global visitsB
    global prevAction
    steps += 1
    state = get_state(obs, pickup, visitsA, visitsB, prevAction)
    if state not in q_table:
        action = np.random.choice(6) # Choose a random action
    else:
        p = softmax(q_table[state])
        action = np.random.choice(6, p=p)
    """if pickup and not (state[0] == 0 and state[1] == 0 and state[7] == 1):
        if action == 5:
            print(state, q_table[state])
            print("jdfisojdf", flush=True)
        while action == 5:
            action = np.random.choice(6)
    if not pickup and not (state[0] == 0 and state[1] == 0 and state[6] == 1):
        while action == 4:
            action = np.random.choice(6)
        '''if np.random.rand() < 0.000:
            action = np.random.choice(6)
        else:
            action = np.argmax(q_table[state])'''"""
    prevAction = action
    #print(state[6], flush=True)
    if action == 4:
        print("P", pickup, flush=True)
    elif action ==5:
        print("D", pickup, flush=True)
    if not pickup:
        at_station = (obs[0] == obs[2] and obs[1] == obs[3]) or (obs[0] == obs[4] and obs[1] == obs[5]) or (obs[0] == obs[6] and obs[1] == obs[7]) or (obs[0] == obs[8] and obs[1] == obs[9])
        at_pickup = state[6] == 1 and at_station
        not_at_pickup = state[6] != 1 and at_station
        action_pickup = action == 4
        if state[0] == 0 and state[1] == 0 and not_at_pickup:
            visitsA += 1
            visitsA = min(visitsA, 3)
        if at_pickup and action_pickup:
            pickup = True
    else:
        at_station = (obs[0] == obs[2] and obs[1] == obs[3]) or (obs[0] == obs[4] and obs[1] == obs[5]) or (obs[0] == obs[6] and obs[1] == obs[7]) or (obs[0] == obs[8] and obs[1] == obs[9])
        at_dest = state[7] == 1 and at_station
        not_at_dest = state[7] != 1 and at_station
        action_drop = action == 5
        if state[0] == 0 and state[1] == 0 and not_at_dest:
            visitsB += 1
            visitsB = min(visitsB, 3)
        if at_dest:
            if action_drop:
                pickup = False
        elif not_at_dest:
            if action_drop:
                pickup = False
                visitsA = 0
                visitsB = 0
    if steps >= 5000:
        pickup = False
        steps = 0
        visitsA = 0
        visitsB = 0
        prevAction = 0
    return action
    # You can submit this random agent to evaluate the performance of a purely random strategy.