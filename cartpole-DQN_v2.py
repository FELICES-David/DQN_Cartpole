# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:35:51 2019

@author: david
"""

##Packages
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

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
NUM_EPISODES = 1000
MEMORY_CAPACITY = 10000
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 1000
TARGET_UPDATE = 10
EDGE_CASE = 0.1
LEARNING_RATE = 0.001


##Q-Network
class DQN(nn.Module):
    
    def __init__(self, inputs, outputs):
        super(DQN,self).__init__()
        self.linear1 = torch.nn.Linear(inputs, 16)
        self.linear2 = torch.nn.Linear(16, 32)
        self.linear3 = torch.nn.Linear(32, outputs)
    
    def forward(self, x):
        x = x.to(device)
        h_relu1 = self.linear1(x).clamp(min=0)
        h_relu2 = self.linear2(h_relu1).clamp(min=0)
        y_pred = self.linear3(h_relu2)
        return y_pred

policy_net = DQN(n_state_values,n_actions).double().to(device)
target_net = DQN(n_state_values,n_actions).double().to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters(),0.001)

##Creating Transition class
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'terminal'))


##Calculating TD-error
def update_Q_target_list(memory,transition):
    Q_target_list = []
    for next_transition in memory:
        if next_transition.state.all() == transition.next_state.all():
            Q_target_list.append(target_net(torch.from_numpy(next_transition.state)).max(0)[0].item())
    if Q_target_list == []:
        return [1]
    return Q_target_list

def Q_target_next_max(memory,transition):
    Q_target_list = update_Q_target_list(memory,transition)
    return max(Q_target_list)

def Q_policy(transition):
    return policy_net(torch.from_numpy(transition.state))[transition.action].item()
    
    
def TD_error(memory,transition):
    r = transition.reward
    if transition.terminal == True:
        R_target = r
    else:
        R_target = r + GAMMA * Q_target_next_max(memory,transition)
    R_policy = Q_policy(transition)
    error = (R_target - R_policy)**2 + EDGE_CASE
    return error


##Replay Memory
class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.error = []
        self.probability = []
        self.position = 0
    
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.probability.append(None)
            self.error.append(None)
        transition = Transition(*args)
        self.memory[self.position] = transition
        self.error[self.position] = TD_error(self.memory,transition)
        sum_errors = sum(self.error)
        for i in range(len(self.error)):
            self.probability[i] = 0
            for j in range(i+1):
                self.probability[i] += self.error[j]/sum_errors
        self.probability[-1] = 1
        self.position = (self.position + 1) % self.capacity
    
    def sample(self,batch_size):
        sample_list = []
        for i in range(batch_size):
            P = random.random()
            index = 0
            while P > self.probability[index]:
                index += 1
            sample_list.append(self.memory[index])
        return sample_list
    
    def __len__(self):
        return len(self.memory)

replay_memory = ReplayMemory(MEMORY_CAPACITY)


##Choosing an action
steps_done = 0

def select_action(state):
    global steps_done
    not_greedy = random.random()
    eps = EPS_END + (EPS_START - EPS_END) * math.exp(-steps_done / EPS_DECAY)
    steps_done += 1
    if not_greedy > eps:
        with torch.no_grad():
            return policy_net(torch.from_numpy(state)).max(0)[1].item()
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long).item()


##Training Loop
episode_durations = []
def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
        

def optimize_model():
    if len(replay_memory) < BATCH_SIZE:
        return
    transition_list = replay_memory.sample(BATCH_SIZE)
    for i in range(BATCH_SIZE):
        state = torch.from_numpy(transition_list[i].state).to(device).double()
        action = torch.tensor([transition_list[i].action]).to(device)
        reward = torch.tensor([transition_list[i].reward]).to(device).double()
        state_action_value = policy_net(state).gather(0,action).max(0)[0].to(device).double()
        next_state = transition_list[i].next_state
        if next_state is None:
            expected_state_action_value = reward
        else:
            next_state = torch.from_numpy(next_state).to(device).double()
            expected_state_action_value = target_net(next_state).max(0)[0] * GAMMA + reward
        expected_state_action_value = expected_state_action_value.to(device).double()
        loss = nn.SmoothL1Loss().cuda()
        actual_loss = loss(state_action_value,expected_state_action_value).to(device).double()
        optimizer.zero_grad()
        actual_loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()


for i_episode in range(NUM_EPISODES):
    state = env.reset()
    for t in count():
        env.render()
        action = select_action(state)
        next_state, reward, done, _ = env.step(action)
        reward = torch.tensor([reward], device=device)
        if done:
            next_state = None
        replay_memory.push(state, action, next_state, reward, done)
        state = next_state
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.close()
plt.ioff()
plt.show()




























            