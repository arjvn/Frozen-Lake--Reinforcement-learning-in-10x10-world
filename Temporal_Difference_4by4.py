import numpy as np
from random import randint
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
from tqdm import tqdm
from time import sleep, strftime
from statistics import mean
import os
from tempfile import TemporaryFile

import argparse
parser = argparse.ArgumentParser(description="enter epsilon, number of episodes and SARSA bool")
parser.add_argument('-TD', '--algorithm', default='True', choices=['True', 'False'],
                    help="True = SARSA, False = Q-Learning. (Default: True: SARSA)")
parser.add_argument('-E', '--epsilon', default='1',
                    help="epsilon value (default: 1)")
parser.add_argument('-Epi', '--episode', default='20000',
                    help="epsilon value (default: 20000)")
parser.add_argument('-a', '--alpha', default='0.1',
                    help="alpha value (default: 0.1)")
parser.add_argument('-g', '--gamma', default='0.9',
                    help="gamma value (default: 0.9)")
args = parser.parse_args()


# EXPLAINATION COMMMENTS IN TEMPORAL DIFFERENCE 10X10. PLS REFER

row = 4
column = 4
no_of_actions_perState = 4
start = (0,0)
goal = (3,3)
traps = [(1,1),(1,3),(2,3),(3,0)]

SARSA = False #(False = Q-Learning)
Gamma = 0.9
epsilon = 1
alpha = 0.1
N_Episodes = 20000

#0: Left, 1: Down, 2: Right, 3: Up
#High Gamma = far sight | Low Gamma = Near sighted
verbose = True

def ComparePolicies(Policy, OldPolicy):
    PolicyChange_Counter = 0
    for x in range(len(Policy)):
        for y in range(len(Policy)):
            if Policy[x][y] != OldPolicy[x][y]:
                PolicyChange_Counter += 1
                # print("PolicyChange_Counter", PolicyChange_Counter)

    return PolicyChange_Counter

def generate_traps():
    check = np.zeros((10,10),int)
    check[0,0] = 1
    i = 0
    while i < 26:
        trap = (randint(0,9), randint(0,9))
        if check[trap[0], trap[1]] == 0:
            check[trap[0], trap[1]] = 1
            traps.append(trap)
            i+=1

def DetermineAction(policy, sa, td, episodes, exploit, explore):
    if td == 1: #E greedy policy
        if np.random.random_sample() < (epsilon*(1 - (episodes/N_Episodes))):
            #explore - randomly select action 0,1,2,3
            action = randint(0, 3)
            # print("#explore - randomly select action 0,1,2,3\n Action:", action)
            explore+=1
            # print("explore:", explore)
        else:
            #exploit - greedy epsilon - select action with highest Q(s,a)
            action = policy[sa[0], sa[1]]
            # print("exploit - greedy epsilon - select action with highest Q(s,a) \n NSA[2]:", action)
            exploit+=1
            # print("exploit:", exploit)


    if td == 0: #greedy policy
            #exploit - greedy epsilon - select action with highest Q(s,a)
            action = policy[sa[0], sa[1]]
            # print("exploit - greedy epsilon - select action with highest Q(s,a) \n NSA[2]:", action)
    return (action, exploit, explore)

def save(steps_perEpi, exploited_steps_perEpi, explored_steps_perEpi, Unique_SA_PairperEpi, Terminal_Reward_perEpi, CumReturnPerEpi, Policy, TrackPolicyChangesPerEpi, CumTrackPolicyChangesPerEpi, alpha, Gamma, epsilon, N_Episodes):
    x_axis = []
    x_axis = np.linspace(0, (N_Episodes-1), N_Episodes)

    docs_dir = os.path.expanduser('~/Documents/Gym/ME5406/TD-4x4/Q_TD-4X4')

    if SARSA == False:
        docs_dir = os.path.expanduser('~/Documents/Gym/ME5406/TD-4x4/Q_TD-4X4')
    elif SARSA == True:
        docs_dir = os.path.expanduser('~/Documents/Gym/ME5406/TD-4x4/S_TD-4X4')

    timestr = strftime("%Y%m%d-%H%M%S")

    # os.mkdir(os.path.join(docs_dir, timestr))
    if not os.path.exists(os.path.join(docs_dir, str(epsilon))):
        os.mkdir(os.path.join(docs_dir, str(epsilon)))

    if not os.path.exists(os.path.join(docs_dir, str(epsilon), str(N_Episodes))):
        os.mkdir(os.path.join(docs_dir, str(epsilon), str(N_Episodes)))

    if not os.path.exists(os.path.join(docs_dir, str(epsilon), str(N_Episodes), timestr)):
        os.mkdir(os.path.join(docs_dir, str(epsilon), str(N_Episodes), timestr))

    dir = os.path.join(docs_dir, str(epsilon), str(N_Episodes))

    constants = []
    constants.append(SARSA)
    constants.append(alpha)
    constants.append(Gamma)
    constants.append(epsilon)
    constants.append(N_Episodes)
    np.save(os.path.join(dir, timestr, 'Constants.npy'), constants)
    np.save(os.path.join(dir, timestr, 'StepsPerEpi.npy'), steps_perEpi)
    np.save(os.path.join(dir, timestr, 'exploited_steps_perEpi.npy'), exploited_steps_perEpi)
    np.save(os.path.join(dir, timestr, 'explored_steps_perEpi.npy'), explored_steps_perEpi)
    np.save(os.path.join(dir, timestr, 'Unique_SA_PairperEpi.npy'), Unique_SA_PairperEpi)
    np.save(os.path.join(dir, timestr, 'Terminal_Reward_perEpi.npy'), Terminal_Reward_perEpi)
    np.save(os.path.join(dir, timestr, 'CumReturnPerEpi.npy'), CumReturnPerEpi)
    np.save(os.path.join(dir, timestr, 'x_axis.npy'), x_axis)
    np.save(os.path.join(dir, timestr, 'Policy.npy'), Policy)
    np.save(os.path.join(dir, timestr, 'TrackPolicyChangesPerEpi.npy'), TrackPolicyChangesPerEpi)
    np.save(os.path.join(dir, timestr, 'CumTrackPolicyChangesPerEpi.npy'), CumTrackPolicyChangesPerEpi)

    print ("directory: {}/{}".format(docs_dir,timestr))

def plot(steps_perEpi,exploited_steps_perEpi,explored_steps_perEpi,Unique_SA_PairperEpi, Terminal_Reward_perEpi, CumReturnPerEpi, TrackPolicyChangesPerEpi, CumTrackPolicyChangesPerEpi):

    #plots
    mpl.style.use('default')
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    x_axis = np.linspace(0, (N_Episodes-1), N_Episodes)

    # Figure 1 Steps each episode
    #ax1.plot(x_axis, steps_perEpi)
    ax1.scatter(x_axis, steps_perEpi, c='C1', s = 10, label='No. of steps', alpha=0.7, edgecolors='none')
    ax1.scatter(x_axis, exploited_steps_perEpi, c='C2', s = 10, label='No. of steps exploited', alpha=0.7, edgecolors='none')
    ax1.scatter(x_axis, explored_steps_perEpi, c='C3', s = 10, label='No. of steps explored', alpha=0.7, edgecolors='none')
    # ax1.scatter(x_axis, Unique_SA_PairperEpi, c='C4', s = 10, label='No. of unique SA pairs', alpha=0.7, edgecolors='none')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    # Figure 2 Cumulative G per episode
    ax2.set_xlabel('No. of episodes')
    ax2.set_ylabel("Average return per Episode")
    ax2.set_ylim([-1,1])
    ax2.plot(x_axis, CumReturnPerEpi)
    ax2.legend(loc='upper right')

    # track policy changes
    ax3.set_xlabel('No. of episodes')
    ax3.set_ylabel("Cumultive changes in policy")
    # ax3.set_ylim([-1,100])
    ax3.plot(x_axis, CumTrackPolicyChangesPerEpi)
    ax3.legend(loc='upper right')
    plt.show()

def drawPathTaken(states_visited, iterator):
    path = np.zeros((row,column),int)
    for i in range(iterator):
        #print("i: {} \naction taken {} \nstates_visited[i][0],states_visited[i][1]]: {} {}".format(i, states_visited[i][2], states_visited[i][0], states_visited[i][1]))
        if states_visited[i][2] == 0:
            path[states_visited[i][0],states_visited[i][1]] = '<-'
        if states_visited[i][2] == 1:
            path[states_visited[i][0],states_visited[i][1]] = '|'
        if states_visited[i][2] == 2:
            path[states_visited[i][0],states_visited[i][1]] = '->'
        if states_visited[i][2] == 3:
            path[states_visited[i][0],states_visited[i][1]] = '|'

def generate_policy_map(input):
    w, h = row, column

    for i in range(w):
        print ("\n")
        for j in range(h):
            if (i,j) == goal:
                print (" \u0489 ", end = '')
            elif  (i,j) in traps:
                print (" \u26B0 ", end = '')
            elif input[i][j] == 0:
                print(" \u2190 ", end = '')
            elif input[i][j] == 1:
                print(" \u2193 ", end = '')
            elif input[i][j] == 2:
                print(" \u2192 ", end = '')
            elif input[i][j] == 3:
                print(" \u2191 ", end = '')
    print("\n")
    return

def UpdatePolicy(Q_SAR, policy):
    for r in range(row):
        for c in range(column):
            best_action_value = float('-inf')
            for action in range(no_of_actions_perState):
                #print("r {}, c {}, action {}".format(r,c,action))
                if Q_SAR[action, r, c] > best_action_value:
                    best_action_value = Q_SAR[action, r, c]
                    #print("\n Q_Avg[action, r, c]:", Q_Avg[action, r, c])
                    Best_action = action
            policy[r,c] = Best_action
                    #print("Best_action:", Best_action)
                #print("policy at {}, {} has best action {}".format(r,c,Best_action))

                #print("policy from best\n", policy)
    return policy

def TransitionFx(x, y, action_taken):

    # print("action_taken:", action_taken)

    if action_taken == 0:
        newstate = [x, y-1]
        #check boundary
    elif action_taken == 1:
        newstate = [x+1 , y]
        #check boundary
    elif action_taken == 2:
        newstate = [x, y+1]
        #check boundary
    elif action_taken == 3:
        newstate = [x-1, y]
        #check boundary
    else:
        raise Exception("action not accounted for")

    if newstate[0] < 0 or newstate[0] > (row-1) or newstate[1] < 0 or newstate[1] > (column-1):
                newstate = [x,y] #bouncy walls

    return (newstate[0], newstate[1])

def TemporalDiffernce(GridOfMap, SARSA):
    # Q_SAR = [[[0 for k in range(no_of_actions_perState)] for j in range(column)] for i in range(row)]
    Q_SAR = np.zeros((no_of_actions_perState, row, column),float)
    # Policy = [[0 for x in range(column)] for y in range(row)]
    Policy = np.zeros((row,column),int)
    episode = 0
    #print ("Begin Monte Carlo")
    # for episode in range(N_Episodes):
    steps_perEpi = []
    exploited_steps_perEpi = []
    explored_steps_perEpi = []
    Terminal_Reward_perEpi = []
    CumReturnPerEpi = []
    Unique_SA_PairperEpi = []
    TrackPolicyChangesPerEpi = []
    CumTrackPolicyChangesPerEpi = []
    CumChangesPolicy = 0

    for episode in tqdm(range(N_Episodes),desc = "Progress : "):
        exploited = 0
        explored = 0
        State = [0,0]
        SA = [0,0,0] # x, y, a
        NSA = [0,0,0] # x, y, a
        SAR_order = []
        step = 0
        SumEpiReward = 0 #for graphing
        CumEpiReturn = [] #for graphing
        ChangesPolicy = 0


        if verbose and episode % (N_Episodes/10) == 0:          #tqmd
            print("EPISODES : {} | ESPSILON : {} | GAMMA : {} | ALPHA : {}".format(episode,epsilon, Gamma, alpha))
            generate_policy_map(Policy)

        while GridOfMap[SA[0]][SA[1]] != 1 and GridOfMap[SA[0]][SA[1]] != -1:


            if step == 0:
                SA[2], exploited, explored = DetermineAction(Policy, SA, 1, episode, exploited, explored)
                step+=1

            NSA[0], NSA[1] = TransitionFx(SA[0],SA[1],SA[2]) #look ahead
            if SARSA == True:
                TD = 1
                NSA[2], exploited, explored = DetermineAction(Policy, NSA, TD, episode, exploited, explored)
                Q_SAR[SA[2], SA[0], SA[1]] = Q_SAR[SA[2], SA[0], SA[1]] + alpha*(GridOfMap[NSA[0], NSA[1]] + (Gamma * Q_SAR[NSA[2], NSA[0], NSA[1]]) - Q_SAR[SA[2], SA[0], SA[1]])
                CumEpiReturn.append(Q_SAR[SA[2], SA[0], SA[1]])
            else:
                # print("QSAR_QL")
                TD = 0
                NSA[2], exploited, explored = DetermineAction(Policy, NSA, TD, episode, exploited, explored) # greedy action

                Q_SAR[SA[2], SA[0], SA[1]] = Q_SAR[SA[2], SA[0], SA[1]] + alpha*(GridOfMap[NSA[0], NSA[1]] + (Gamma * Q_SAR[NSA[2], NSA[0], NSA[1]]) - Q_SAR[SA[2], SA[0], SA[1]])
                CumEpiReturn.append(Q_SAR[SA[2], SA[0], SA[1]])
                TD = 1
                NSA[2], exploited, explored = DetermineAction(Policy, SA, TD, episode, exploited, explored) #E greedy action


            SA = copy.copy(NSA)
            step +=1




        steps_perEpi.append(step) # for graphing
        explored_steps_perEpi.append(explored) # for graphing
        exploited_steps_perEpi.append(exploited) # for graphing
        OldPolicy = copy.deepcopy(Policy)

        Policy = UpdatePolicy(Q_SAR, Policy)

        ChangesPolicy = ComparePolicies(Policy, OldPolicy)
        CumChangesPolicy += ChangesPolicy
        TrackPolicyChangesPerEpi.append(ChangesPolicy)
        CumTrackPolicyChangesPerEpi.append(CumChangesPolicy)


    print(GridOfMap)
    generate_policy_map(Policy)
    # plot(steps_perEpi, exploited_steps_perEpi, explored_steps_perEpi, Unique_SA_PairperEpi, Terminal_Reward_perEpi, CumReturnPerEpi, TrackPolicyChangesPerEpi, CumTrackPolicyChangesPerEpi)
    # save(steps_perEpi, exploited_steps_perEpi, explored_steps_perEpi, Unique_SA_PairperEpi, Terminal_Reward_perEpi, CumReturnPerEpi, Policy, TrackPolicyChangesPerEpi, CumTrackPolicyChangesPerEpi, alpha, Gamma, epsilon, N_Episodes)
    return

def main():
    #GenerateMap
    MapGrid = np.zeros((row,column),int)
    for x,y in traps:
        MapGrid[x][y] = -1
    MapGrid[goal[0],goal[1]] = 1 #mapgrid is only for reference. it should not be changed after this

    print(MapGrid)
    # sleep(10)
    TemporalDiffernce(MapGrid, SARSA) #play game + episode iteration
    if SARSA == False:
        print("METHOD: Q_Learning, EPSILON:{}, NO.EPISODES: {}, GAMMA: {}, ALPHA: {}".format(epsilon, N_Episodes, Gamma, alpha))
    else:
        print("METHOD: SARSA, EPSILON:{}, NO.EPISODES: {}, GAMMA: {}, ALPHA: {}".format(epsilon, N_Episodes, Gamma, alpha))
if __name__== "__main__":

    SARSA = args.algorithm #(False = Q-Learning)
    Gamma = args.gamma
    Gamma = float(Gamma)
    epsilon = args.epsilon
    epsilon = float(epsilon)
    alpha = args.alpha
    alpha = float(alpha)
    N_Episodes = args.episode  #add stopping coniditon instead
    N_Episodes = int(N_Episodes)
    main()
