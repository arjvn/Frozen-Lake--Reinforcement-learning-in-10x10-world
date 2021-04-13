import numpy as np
from random import randint
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from time import strftime
import os
import copy

# EXPLAINATION COMMMENTS IN TEMPORAL DIFFERENCE 10X10. PLS REFER

row = 4
column = 4
#MapGrid[row][column]
no_of_actions_perState = 4
start = (0,0)
goal = (3,3)
traps = [(1,1),(1,3),(2,3),(3,0)]

Gamma = 0.5 #should be 0.5ish no??
N_Episodes = 20000      #add stopping coniditon instead
epsilon = 0.7

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


def plot(steps_perEpi,exploited_steps_perEpi,explored_steps_perEpi,Unique_SA_PairperEpi, Terminal_Reward_perEpi, Average_return_perEpi, CumTrackPolicyChangesPerEpi):

    #plots
    mpl.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2)
    x_axis = np.linspace(0, (N_Episodes-1), N_Episodes)
    # Figure 1 Steps each episode
    #ax1.plot(x_axis, steps_perEpi)
    ax1.scatter(x_axis, steps_perEpi, c='C1', s = 10, label='No. of steps', alpha=0.7, edgecolors='none')
    ax1.scatter(x_axis, exploited_steps_perEpi, c='C2', s = 10, label='No. of steps exploited', alpha=0.7, edgecolors='none')
    ax1.scatter(x_axis, explored_steps_perEpi, c='C3', s = 10, label='No. of steps explored', alpha=0.7, edgecolors='none')
    ax1.scatter(x_axis, Unique_SA_PairperEpi, c='C4', s = 10, label='No. of unique SA pairs', alpha=0.7, edgecolors='none')
    ax1.legend()
    ax1.grid(True)

    # Figure 2 Cumulative G per episode
    ax2.set_xlabel = ('No. of episodes')
    ax2.set_ylabel = ("Cum Changes in Policy")
    # ax2.set_ylim([-1,1])
    ax2.plot(CumTrackPolicyChangesPerEpi)
    plt.show()

def save(steps_perEpi, exploited_steps_perEpi, explored_steps_perEpi, Unique_SA_PairperEpi, Terminal_Reward_perEpi, CumReturnPerEpi, Policy, TrackPolicyChangesPerEpi, CumTrackPolicyChangesPerEpi, Gamma, epsilon, N_Episodes):
    x_axis = []
    x_axis = np.linspace(0, (N_Episodes-1), N_Episodes)

    docs_dir = os.path.expanduser('~/Documents/Gym/ME5406/MC-4x4')

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

    print ("directory: {}/{}".format(dir,timestr))

def drawPathTaken(states_visited, iterator):
    path = np.zeros((4,4),int)
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
    w, h = 4, 4
    r = 0

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
            r+=1
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
    Policy = np.zeros((4,4),int)
    episode = 0
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
                #print("#explore - randomly select action 0,1,2,3\n State:", State)
                action = randint(0, 3)
                OldState = State
                State = TransitionFx(State, action) #new state
                #print("New State:", State)
                explored +=1
            else:
                #exploit - greedy epsilon - select action with highest Q(s,a)
                #print("exploit - greedy epsilon - select action with highest Q(s,a) \n State:", State)
                action = Policy[State[0], State[1]]
                OldState = State
                State = TransitionFx(State, action) #new state
                #print("New State:", State)
                exploited+=1

            temp = [OldState[0], OldState[1], action, GridOfMap[State[0], State[1]]]

            if checkboxFirstVisit[action, OldState[0], OldState[1]] == 0:
                SAR_order.append(temp)
                checkboxFirstVisit[action, OldState[0], OldState[1]] = 1
                Unique_SA_PairCounter+=1 # for graphing
                #print(checkboxFirstVisit)
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
        #print("Q_Average matrix:\n", Q_SAR_avg)
        OldPolicy = copy.deepcopy(Policy)

        Policy = UpdatePolicy(Q_SAR_avg, Policy)

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
    CumTrackPolicyChangesPerEpi = [x / CumTrackPolicyChangesPerEpi[-1] for x in CumTrackPolicyChangesPerEpi]
    # plot(steps_perEpi, exploited_steps_perEpi, explored_steps_perEpi, Unique_SA_PairperEpi, Terminal_Reward_perEpi, Average_return_perEpi, CumTrackPolicyChangesPerEpi)
    # save(steps_perEpi, exploited_steps_perEpi, explored_steps_perEpi, Unique_SA_PairperEpi, Terminal_Reward_perEpi, CumReturnPerEpi, Policy, TrackPolicyChangesPerEpi, CumTrackPolicyChangesPerEpi, Gamma, epsilon, N_Episodes)
    return

def main():
    #GenerateMap
    MapGrid = [[0 for x in range(column)] for y in range(row)]
    MapGrid = np.zeros((4,4),int)
    for x,y in traps:
        MapGrid[x][y] = -1
    MapGrid[goal[0],goal[1]] = 1 #mapgrid is only for reference. it should not be changed after this

    MonteCarlo(MapGrid) #play game + episode iteration
    print("METHOD: Monte Carlo, EPSILON:{}, NO.EPISODES: {}, GAMMA: {}".format(epsilon, N_Episodes, Gamma))

if __name__== "__main__":
  main()
