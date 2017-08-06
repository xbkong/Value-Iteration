import numpy as np


# The function gridWorld returns the transition function T and reward function M
def gridWorld():

    # Grid world layout:
    #
    #  ---------------------
    #  |  0 |  1 |  2 |  3 |
    #  ---------------------
    #  |  4 |  5 |  6 |  7 |
    #  ---------------------
    #  |  8 |  9 | 10 | 11 |
    #  ---------------------
    #  | 12 | 13 | 14 | 15 |
    #  ---------------------
    #
    #  Goal state: 15 
    #  Bad state: 9
    #  End state: 16
    #
    #  The end state is an absorbing state that the agent transitions 
    #  to after visiting the goal state.
    #
    #  There are 17 states in total (including the end state) 
    #  and 4 actions (up, down, right, left).
    #

    #%%%%%%%%%%%%%%%%%%% rewards %%%%%%%%%%%%%%%%%%%%%%

    # Rewards are stored in a one dimensional array R[s]
    #
    # All states have a reward of -1 except:
    # Goal state: 100
    # Bad state: -70
    # End state: 0 

    # initialize rewards to -1
    R = -np.ones((17))

    # set rewards
    R[15] = 100  # goal state
    R[9] = -70   # bad state
    R[16] = 0    # end state

    return R

up = 0
down = 1
left = 2
right = 3


# Grid world layout:
#
#  ---------------------
#  |  0 |  1 |  2 |  3 |
#  ---------------------
#  |  4 |  5 |  6 |  7 |
#  ---------------------
#  |  8 |  9 | 10 | 11 |
#  ---------------------
#  | 12 | 13 | 14 | 15 |
#  ---------------------

def get_new_state(init_s, direction):
    new_s = init_s
    if direction == up and init_s - 4 >= 0:
        new_s = init_s - 4
    elif direction == down and init_s + 4 < 16:
        new_s = init_s + 4
    elif direction == left and init_s not in [0, 4, 8, 12]:
        new_s = init_s - 1
    elif direction == right and init_s not in [3, 7, 11]:
        new_s = init_s + 1
    return new_s


def get_new_stage_and_probs(init_s, direction, a, b):
    if init_s == 15:
        return [[16, 1]]
    if direction == left or direction == right:
        return [[get_new_state(init_s, direction), a],
                [get_new_state(init_s, up), b],
                [get_new_state(init_s, down), b]]
    if direction == up or direction == down:
        return [[get_new_state(init_s, direction), a],
                [get_new_state(init_s, left), b],
                [get_new_state(init_s, right), b]]


def value_iteration(a=0.8, b=0.1):
    R = gridWorld()
    discount = 0.99
    values = [[] for i in range(17)]
    actions = [[] for i in range(17)]

    for i in range(17):
        values[i] += [R[i]]

    for t in range(1, 40):
        terminate = True
        for i in range(17):
            if i == 16:
                values[i] += [0]
                actions[i] += [None]
                break
            maxreward = -1*float('inf')
            bestaction = None
            for action in [up, down, left, right]:
                actionsum = 0
                for transition in get_new_stage_and_probs(i, action, a, b):
                    prob = transition[1]
                    new_state = transition[0]
                    reward = R[i]
                    actionsum += prob * (reward + discount * values[new_state][t-1])
                if actionsum >= maxreward:
                    maxreward = actionsum
                    bestaction = action
            values[i] += [maxreward]
            if values[i][t] - values[i][t-1] > 0.01:
                terminate = False
            actions[i] += [bestaction]
        if terminate: #terminate check
            break

    values = np.array(values)
    actions = np.array(actions)
    import pandas as pd
    pd.set_option('display.max_columns', 28)
    pd.set_option('display.max_rows', 28)
    print pd.DataFrame(values)
    print pd.DataFrame(actions)
    pd.reset_option('display.max_columns')


value_iteration(a=0.9, b=0.05)
