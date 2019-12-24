# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:09:37 2019

@author: david
"""

##Packages
import gym
import random
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim


##Environment
env = gym.make('CartPole-v1')

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
    
plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_actions = env.action_space.n
n_state_values = 4


##Hyperparamaters
NUM_EPISODES = 2000
MEMORY_CAPACITY = 30000
BATCH_SIZE = 128
GAMMA = 0.998
EPS_START = 1
EPS_END = 0.05
EPS_DECAY = 1000
TARGET_UPDATE = 20
PLOT_UPDATE = 20
LEARNING_RATE = 0.001
SOFT_UPDATE = 0.1
SUCCESS = 100
TRIALS_MAX = 10
EVALUATIONS = 100
PERCENTILE_HIGH = 85
PERCENTILE_LOW = 0



##Q-Network
class DQN(nn.Module):
    
    def __init__(self, inputs, outputs):
        super(DQN,self).__init__()
        self.linear1 = nn.Linear(inputs, 32)
        self.linear4 = nn.Linear(32, outputs)
        self.dropout =  nn.Dropout(p=0.25)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear4(x)
        return x

##Creating Transition class
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'terminal')) ##everything needs to be tensors

##Simplifying Inputs
def state_simplify(state_array):
    state_array[0] = round(state_array[0],1)
    state_array[1] = round(state_array[1],0)
    state_array[2] = round(state_array[2],2)
    state_array[3] = round(state_array[3],1)
    state_tensor = torch.from_numpy(np.array([state_array])).float().to(device)
    return state_tensor

##Replay Memory
class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        transition = Transition(*args)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity
    
    def sample(self,batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)



class ReplayMemory2(object):
    
    def __init__(self, episode_capacity):
        self.episode_capacity = episode_capacity
        self.memory = [[]]
        self.episode = 0
        self.percentile = 0
    
    def push(self, *args):
        transition = Transition(*args)
        self.memory[self.episode].append(transition)
        if transition.terminal:
            self.episode = (self.episode + 1) % self.episode_capacity
            if len(self.memory) < self.episode_capacity:
                self.memory.append([])
            else:
                self.memory[self.episode] = []
    
    def __len__(self):
        return len(self.memory)
    
    def len_episodes(self):
        list_len = []
        for episode in self.memory:
            list_len.append(len(episode))
        return list_len, np.percentile(list_len, PERCENTILE_LOW), np.percentile(list_len, PERCENTILE_HIGH)
    
    def sample(self,batch_size):
        sample_list=[]
        list_len, percentile_low, percentile_high = self.len_episodes()
        for i_episode in range(len(self.memory)):
            if list_len[i_episode] >= percentile_high or list_len[i_episode] < percentile_low:
                sample_list += self.memory[i_episode]
        return random.sample(sample_list, batch_size)
    


Training = 'Failure'
Trials = 0
while Training == 'Failure'and Trials < TRIALS_MAX:
    Trials += 1
    
    policy_net = DQN(n_state_values,n_actions).float().to(device)
    target_net = DQN(n_state_values,n_actions).float().to(device)
    
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.RMSprop(policy_net.parameters(),LEARNING_RATE)
    
    replay_memory = ReplayMemory(MEMORY_CAPACITY)
    memory_size = []
    steps_done = 0
    
    ##Choosing an action
    def select_action(state):
        global steps_done
        not_greedy = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * steps_done / EPS_DECAY)
        steps_done += 1
        if not_greedy > eps_threshold:
            with torch.no_grad():
                return policy_net(state).argmax().item()
        else:
            return torch.tensor([random.randrange(n_actions)], device=device, dtype=torch.long).item()
    
    
    ##Training Loop
    means = [0]
    means_max = 0
    episode_durations = []
    
    
    def plot_durations():
        global means
        global means_max
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(episode_durations, dtype=torch.float)
        plt.title('Training...' + str(Trials))
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        for i in range(PLOT_UPDATE):
            memory_size.append(len(replay_memory)/(30000/100))
        plt.plot(memory_size)
        if len(durations_t) >= 50:
            means = durations_t.unfold(0, 50, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(49), means))
            means_max = max(means)
            plt.plot(means.numpy())
        plt.pause(0.001)
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())
            
            
    def optimize_model():
        if len(replay_memory) < BATCH_SIZE:
            return
        transition_list = replay_memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transition_list))
        batch_state = torch.cat(batch.state).to(device)
        batch_action = torch.cat(batch.action).to(device)
        batch_reward = torch.cat(batch.reward).to(device)
        batch_state_action_value = policy_net(batch_state).gather(1,batch_action).to(device)
        batch_next_state = torch.cat(batch.next_state).to(device)
        batch_expected_state_action_value = target_net(batch_next_state).to(device)
        batch_expected_state_action_value = batch_expected_state_action_value.max() * GAMMA + batch_reward
        batch_expected_state_action_value = batch_expected_state_action_value.to(device)
        loss = nn.SmoothL1Loss().cuda()
        actual_loss = loss(batch_state_action_value,batch_expected_state_action_value).to(device)
        optimizer.zero_grad()
        actual_loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()


    for i_episode in range(NUM_EPISODES):
        state = env.reset()
        state = state_simplify(state)
        for t in count():
            action = select_action(state)
            next_state, reward, done, _ = env.step(action)
            action = torch.tensor([[action]], device=device)
            reward = torch.tensor([[reward]], device=device) * (1 - done)
            next_state = state = state_simplify(next_state)
            replay_memory.push(state, action, next_state, reward, done)
            state = next_state
            optimize_model()
            if done:
                episode_durations.append(t + 1)
                if i_episode % PLOT_UPDATE == PLOT_UPDATE - 1:
                    plot_durations()
                break
        if i_episode % TARGET_UPDATE == 0:    
            for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
                target_param.data.copy_(SOFT_UPDATE*policy_param.data + (1.0-SOFT_UPDATE)*target_param.data)
        if means[-1] >= SUCCESS:
            print('Success')
            Training = 'Success'
            break
        if means[-1] < means_max * 0.2:
            print('Failure')
            break

print('Complete')
env.close()
plt.ioff()
plt.show()

## Evaluation
def evaluation(number_of_tests):
    global best_test
    global all_tests
    global mean_reward
    max_reward = 0
    best_test = [0, []]
    all_tests = []
    mean_reward = 0
    for i in range(number_of_tests):
        state = env.reset()
        state = state_simplify(state)
        cumulative_reward = 0
        current_test = [cumulative_reward, [state]] 
        for t in count():
            env.render()
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            cumulative_reward += reward
            state = state_simplify(state)
            current_test[0] = cumulative_reward
            current_test[1].append(state)
            if done:
                all_tests.append(current_test)
                if cumulative_reward > max_reward:
                    max_reward = cumulative_reward
                    best_test = current_test
                break
    env.close()
    for i in range(number_of_tests):
        mean_reward += all_tests[i][0]/number_of_tests
    return all_tests, best_test, mean_reward

if Training == 'Success':
    evaluation(EVALUATIONS)
    print(best_test)
    print(mean_reward)