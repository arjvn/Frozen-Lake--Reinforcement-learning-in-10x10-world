import numpy as np
from random import randint
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from time import strftime
import os
import copy

import argparse
parser = argparse.ArgumentParser(description="enter epsilon, number of episodes and gamma")
parser.add_argument('-E', '--epsilon', default='1',
                    help="epsilon value (default: 1)")
parser.add_argument('-Epi', '--episode', default='1000000',
                    help="epsilon value (default: 1000000)")
parser.add_argument('-g', '--gamma', default='0.9',
                    help="gamma value (default: 0.9)")
args = parser.parse_args()


# GRID SIZE CAN BE CHANGED FOR EXPERIMENTATION - ACTIVATE GENERATE TRAPS
row = 10
column = 10
#MapGrid[row][column]
no_of_actions_perState = 4
start = (0,0)
goal = (row-1,column-1)
# traps = [] # uncomment this line and comment next line if you want to generate a random map.
#function generate_traps needs to be activated in main if you want random generation of traps
traps = [(0,4),(0,9),(1,3),(1,5),(1,6),(2,0),(2,4),(3,0),(3,4),(3,8),(5,1),(5,4),(5,9),(6,9),(7,1),(7,3),(7,5),(8,0),(8,3),(8,4),(9,1),(9,3),(9,5),(9,8)]

#0: Left, 1: Down, 2: Right, 3: Up
#High Gamma = far sight | Low Gamma = Near sighted
verbose = True

def ComparePolicies(Policy, OldPolicy):
    PolicyChange_Counter = 0
    for x in range(len(Policy)):
        for y in range(len(Policy)):
            if Policy[x][y] != OldPolicy[x][y]:
                PolicyChange_Counter += 1

    return PolicyChange_Counter

# PRODUCES A BASIC PLOT - DOES NOT USE ROLLING AVERAGE - ONLY USED TO VALIDATE TAHT DATA COLLECTED IS CORRECTED NOT USED IN PAPER
def plot(steps_perEpi,exploited_steps_perEpi,explored_steps_perEpi,Unique_SA_PairperEpi, Terminal_Reward_perEpi, Average_return_perEpi, CumReturnPerEpi):

    #plots
    mpl.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2)
    x_axis = np.linspace(0, (N_Episodes-1), N_Episodes)
    ax1.scatter(x_axis, steps_perEpi, c='C1', s = 10, label='No. of steps', alpha=0.7, edgecolors='none')
    ax1.scatter(x_axis, exploited_steps_perEpi, c='C2', s = 10, label='No. of steps exploited', alpha=0.7, edgecolors='none')
    ax1.scatter(x_axis, explored_steps_perEpi, c='C3', s = 10, label='No. of steps explored', alpha=0.7, edgecolors='none')
    ax1.scatter(x_axis, Unique_SA_PairperEpi, c='C4', s = 10, label='No. of unique SA pairs', alpha=0.7, edgecolors='none')
    ax1.legend()
    ax1.grid(True)

    # Figure 2 Cumulative G per episode
    ax2.set_xlabel('No. of episodes')
    ax2.set_ylabel("CumReturnPerEpi")
    ax2.set_ylim([-1,1])
    ax2.plot(CumReturnPerEpi)

    # ax.legend()
    plt.show()

# SAVE FUNCTION IS FOR LOGGING - IT WILL CREATE A DIRECTORY IN YOUR DOCUMENTS - DEACTIVATED FOR SUBMISSION. ACTIVATE AT LINE: 327
def save(steps_perEpi, exploited_steps_perEpi, explored_steps_perEpi, Unique_SA_PairperEpi, Terminal_Reward_perEpi, Average_return_perEpi, CumReturnPerEpi, Policy):
    print("saving")
    x_axis = []
    x_axis = np.linspace(0, (N_Episodes-1), N_Episodes)

    docs_dir = os.path.expanduser('~/Documents/Gym/ME5406/MC-10x10')

    timestr = strftime("%Y%m%d-%H%M%S")

    # os.mkdir(os.path.join(docs_dir, timestr))
    if not os.path.exists(os.path.join(docs_dir, str(epsilon))):
        os.mkdir(os.path.join(docs_dir, str(epsilon)))

    if not os.path.exists(os.path.join(docs_dir, str(epsilon), str(N_Episodes))):
        os.mkdir(os.path.join(docs_dir, str(epsilon), str(N_Episodes)))

    if not os.path.exists(os.path.join(docs_dir, str(epsilon), str(N_Episodes), timestr)):
        os.mkdir(os.path.join(docs_dir, str(epsilon), str(N_Episodes), timestr))

    dir = os.path.join(docs_dir, str(epsilon), str(N_Episodes))

    np.save(os.path.join(dir, timestr, 'StepsPerEpi.npy'), steps_perEpi)
    np.save(os.path.join(dir, timestr, 'exploited_steps_perEpi.npy'), exploited_steps_perEpi)
    np.save(os.path.join(dir, timestr, 'explored_steps_perEpi.npy'), explored_steps_perEpi)
    np.save(os.path.join(dir, timestr, 'Unique_SA_PairperEpi.npy'), Unique_SA_PairperEpi)
    np.save(os.path.join(dir, timestr, 'Terminal_Reward_perEpi.npy'), Terminal_Reward_perEpi)
    np.save(os.path.join(dir, timestr, 'CumReturnPerEpi.npy'), CumReturnPerEpi)
    np.save(os.path.join(dir, timestr, 'Average_return_perEpi.npy'), Average_return_perEpi)
    np.save(os.path.join(dir, timestr, 'x_axis.npy'), x_axis)
    np.save(os.path.join(dir, timestr, 'Policy.npy'), Policy)
    print ("directory: {}/{}".format(docs_dir,timestr))

def generate_traps():
    check = np.zeros((10,10),int)
    check[0,0] = 1
    i = 0
    while i < ((row*column)/4):
        trap = (randint(0,row-1), randint(0,column-1))
        if check[trap[0], trap[1]] == 0:
            check[trap[0], trap[1]] = 1
            traps.append(trap)
            i+=1

def generate_policy_map(input):


    for i in range(row):
        print ("\n")
        for j in range(column):
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

def UpdatePolicy(Q_Avg, policy):

    for r in range(row):
        for c in range(column):
            best_action_value = float('-inf')
            for action in range(no_of_actions_perState):
                #print("r {}, c {}, action {}".format(r,c,action))
                if Q_Avg[action, r, c] > best_action_value:
                    best_action_value = Q_Avg[action, r, c]
                    #print("\n Q_Avg[action, r, c]:", Q_Avg[action, r, c])
                    Best_action = action
            policy[r,c] = Best_action
                    #print("Best_action:", Best_action)
                #print("policy at {}, {} has best action {}".format(r,c,Best_action))

                #print("policy from best\n", policy)
    return policy

def TransitionFx(state, action_taken):

    # print("action_taken:", action_taken)

    if action_taken == 0:
        newstate = [state[0], state[1]-1]
        #check boundary
    elif action_taken == 1:
        newstate = [state[0]+1 , state[1]]
        #check boundary
    elif action_taken == 2:
        newstate = [state[0], state[1]+1]
        #check boundary
    elif action_taken == 3:
        newstate = [state[0]-1, state[1]]
        #check boundary
    else:
        raise Exception("action not accounted for")

    if newstate[0] < 0 or newstate[0] > (row-1) or newstate[1] < 0 or newstate[1] > (column-1):
                newstate = state #bouncy walls

    return (newstate)

def MonteCarlo(GridOfMap):
    Q_SAR = np.zeros((N_Episodes, no_of_actions_perState, row, column),float)
    Policy = np.zeros((row,column),int)
    episode = 0
    #print ("Begin Monte Carlo")
    # for episode in range(N_Episodes):
    steps_perEpi = []
    exploited_steps_perEpi = []
    explored_steps_perEpi = []
    Terminal_Reward_perEpi = []
    Average_return_perEpi = []
    Unique_SA_PairperEpi = []
    CumReturnPerEpi = []
    TrackPolicyChangesPerEpi = []
    CumTrackPolicyChangesPerEpi = []
    CumChangesPolicy = 0

    for episode in tqdm(range(N_Episodes),desc = "Progress : "):
        exploited = 0
        explored = 0
        ops_epsilon = epsilon
        State = [0,0]
        SAR_order = []
        step = 0
        checkboxFirstVisit = np.zeros((no_of_actions_perState, row, column),int)
        SumEpiReward = 0
        AverageEpiReturn = 0
        Unique_SA_PairCounter = 0
        while GridOfMap[State[0]][State[1]] != 1 and GridOfMap[State[0]][State[1]] != -1:

            if np.random.random_sample() < ops_epsilon:
                #explore - randomly select action 0,1,2,3
                action = randint(0, 3)
                OldState = State
                State = TransitionFx(State, action) #new state
                explored +=1
            else:
                #exploit - greedy epsilon - select action with highest Q(s,a)
                action = Policy[State[0], State[1]]
                OldState = State
                State = TransitionFx(State, action) #new state
                exploited+=1

            temp = [OldState[0], OldState[1], action, GridOfMap[State[0], State[1]]]

            if checkboxFirstVisit[action, OldState[0], OldState[1]] == 0:
                SAR_order.append(temp)
                checkboxFirstVisit[action, OldState[0], OldState[1]] = 1
                Unique_SA_PairCounter+=1 # for graphing
            step+=1

        Terminal_Reward = GridOfMap[State[0],State[1]]
        Terminal_Reward_perEpi.append(Terminal_Reward) # for graphing
        SARreversed = SAR_order[::-1]
        i=0
        ReturnFromNextStep = Terminal_Reward
        for r,c,a,rw in SARreversed:            #[State[0], State[1], action, state_reward]
            if i==0:
                Q_SAR[episode, a, r, c] = Terminal_Reward
                SumEpiReward+=Q_SAR[episode, a, r, c] # for graphing
            else:

                Q_SAR[episode, a, r, c] = rw + (Gamma*ReturnFromNextStep)
                SumEpiReward+=Q_SAR[episode, a, r, c] # for graphing
                ReturnFromNextStep = Q_SAR[episode, a, r, c]
            i+=1

        AverageEpiReturn = SumEpiReward/step # for graphing
        Average_return_perEpi.append(AverageEpiReturn) # for graphing
        CumReturnPerEpi.append(SumEpiReward)
        steps_perEpi.append(step) # for graphing
        explored_steps_perEpi.append(explored) # for graphing
        exploited_steps_perEpi.append(exploited) # for graphing
        Unique_SA_PairperEpi.append(Unique_SA_PairCounter) #for graphing
        Q_SAR_avg = np.average(Q_SAR, axis = 0)
        OldPolicy = copy.deepcopy(Policy)
        Policy = UpdatePolicy(Q_SAR_avg, Policy)
        Policy[0][0] = 1
        ChangesPolicy = ComparePolicies(Policy, OldPolicy)
        CumChangesPolicy += ChangesPolicy
        TrackPolicyChangesPerEpi.append(ChangesPolicy)
        CumTrackPolicyChangesPerEpi.append(CumChangesPolicy)
        #print("\n EPISODES : {} | EXPLOIED : {} | EXPLORED : {}".format(episode,exploited, explored))


        if verbose and episode % (N_Episodes/10) == 0:          #tqmd
            generate_policy_map(Policy)
            print("METHOD: Monte Carlo, EPSILON:{}, NO.EPISODES: {}, GAMMA: {}".format(epsilon, N_Episodes, Gamma))


    print(GridOfMap)
    generate_policy_map(Policy)
    plot(steps_perEpi, exploited_steps_perEpi, explored_steps_perEpi, Unique_SA_PairperEpi, Terminal_Reward_perEpi, Average_return_perEpi, CumReturnPerEpi)
    # save(steps_perEpi, exploited_steps_perEpi, explored_steps_perEpi, Unique_SA_PairperEpi, Terminal_Reward_perEpi, Average_return_perEpi, CumReturnPerEpi, Policy)
    return

def main():
    #GenerateMap
    MapGrid = np.zeros((row,column),int)
    # generate_traps()
    for x,y in traps:
        MapGrid[x][y] = -1
    MapGrid[goal[0],goal[1]] = 1 #mapgrid is only for reference. it should not be changed after this
    print(MapGrid)
    # sleep(5)
    MonteCarlo(MapGrid) #play game + episode iteration
    print("METHOD: Monte Carlo, EPSILON:{}, NO.EPISODES: {}, GAMMA: {}".format(epsilon, N_Episodes, Gamma))

if __name__== "__main__":
    Gamma = args.gamma
    Gamma = float(Gamma)
    epsilon = args.epsilon
    epsilon = float(epsilon)
    N_Episodes = args.episode  #add stopping coniditon instead
    N_Episodes = int(N_Episodes)
    main()
