# To run Sarsa algorithm, run:
#python 113554829_project1_SARSA.py -s sarsa
#To run Sarsa(λ) algorithm, run:
#python 113554829_project1_SARSA.py -s sarsa_lambda

import string
from argparse import ArgumentParser
import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def distance_calculator(path):
    distance = 0
    for i in range(len(path) - 1):
        distance += D[path[i]][path[i + 1]]
    return distance

def stochastic_factor(current_state, q, epsilon):
    # find the potential next states(actions) for current state
    nstate_possible = np.where(np.array(D[current_state]) > 0)[0]
    if np.random.rand() > epsilon:  # greedy
        q_of_next_states = q[current_state][nstate_possible]
        next_state = nstate_possible[np.argmax(q_of_next_states)]
    else: # random select
        next_state = random.choice(nstate_possible)
    return next_state


def sarsa(start_state=3, episodes=10000, gamma=0.8, epsilon=0.8, alpha=0.01):
    print("-" * 20)
    print("sarsa begins ...")
    if start_state == 0:
        raise Exception("start node(state) can't be target node(state)!")
    total_path_Length = []
    # init all q(s,a)
    q = np.zeros((num_nodes, num_nodes))  # num_states * num_actions
    for i in range(1, episodes + 1):
        s_cur = start_state
        next_state = stochastic_factor(s_cur, q, epsilon=epsilon)
        path = [s_cur]
        len_of_path = 0
        while True:
            s_next_next = stochastic_factor(next_state, q, epsilon=epsilon)
            # update q
            reward = -D[s_cur][next_state]
            #print(reward)
            delta = reward + gamma * q[next_state, s_next_next] - q[s_cur, next_state]
            q[s_cur, next_state] = q[s_cur, next_state] + alpha * delta
            # update current state
            s_cur = next_state
            next_state = s_next_next
            len_of_path += -reward
            path.append(s_cur)
            if s_cur == 0:
                break
        total_path_Length.append(len_of_path)
        list2=[]
        list2.append(distance_calculator(path))
    strs = "best path for node {} to node 0: ".format(start_state)
    strs += "->".join([str(i) for i in path])
    print(strs)
    return list2

def sarsa_lambda(start_state=3, episodes=10000, gamma=0.8, epsilon=0.8, alpha=0.01, lamda=0.9):
    print("-" * 20)
    print("sarsa(lamda) begins ...")
    if start_state == 0:
        raise Exception("start node(state) can't be target node(state)!")
    total_path_Length = []
    # init all q(s,a)
    q = np.zeros((num_nodes, num_nodes))  # num_states * num_actions
    for i in range(1, episodes + 1):
        s_cur = start_state
        next_state = stochastic_factor(s_cur, q, epsilon=epsilon)
        e = np.zeros((num_nodes, num_nodes))  # eligibility traces
        path = [s_cur]  # save the path for every event
        len_of_path = 0
        while True:
            s_next_next = stochastic_factor(next_state, q, epsilon=epsilon)
            # update q
            e[s_cur, next_state] = e[s_cur, next_state] + 1
            reward = -D[s_cur][next_state]
            delta = reward + gamma * q[next_state, s_next_next] - q[s_cur, next_state]
            q = q + alpha * delta * e
            # update e
            e = gamma * lamda * e
            # update current state
            s_cur = next_state
            next_state = s_next_next
            len_of_path += -reward
            path.append(s_cur)
            if s_cur == 0:  # if current state is target state, finish the current event
                break
        total_path_Length.append(len_of_path)
        list2=[]
        list2.append(distance_calculator(path))
    # print the best path for start state to target state
    strs = "best path for node {} to node 0: ".format(start_state)
    strs += "->".join([str(i) for i in path])
    print(strs)
    return list2

if __name__ == '__main__':
    # adjacent matrix
    # the target node is 0
    D = [[0, 4, 0, 0, 0, 0, 0, 8, 0],
         [4, 0, 8, 0, 0, 0, 0, 9, 0],
         [0, 8, 0, 7, 0, 4, 0, 0, 3],
         [0, 0, 7, 0, 9, 6, 0, 0, 0],
         [0, 0, 0, 9, 0, 9, 0, 0, 0],
         [0, 0, 4, 6, 9, 0, 3, 0, 0],
         [0, 0, 0, 0, 0, 3, 0, 3, 4],
         [8, 9, 0, 0, 0, 0, 3, 0, 5],
         [0, 0, 3, 0, 0, 0, 4, 5, 0]]
    num_nodes = len(D)
    parser = ArgumentParser()
    parser.add_argument("-s", "--solution", help="select the solution", type=str, default="sarsa")
    args = parser.parse_args()
    solution = args.solution
    if solution == "sarsa" or solution == "vi":
        vari1=[]
        for i in range(100):
            var1 = sarsa(start_state=3, episodes=10000, gamma=0.8, epsilon=0.8, alpha=0.01)
            vari1.append(np.average(var1))
        plt.plot(vari1)
        plt.show()
    elif solution == "sarsa(lambda)" or solution == "sarsa_lambda":
        vari1=[]
        for i in range(100):
            var1 = sarsa_lambda(start_state=3, episodes=10000, gamma=0.8, epsilon=0.8, alpha=0.01, lamda=0.9)
            vari1.append(np.average(var1))
        plt.plot(vari1)
        plt.show()
    else:
        print("solution has not been realized!")