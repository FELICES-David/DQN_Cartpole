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
environment = gym.make('CartPole-v1')

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
    
plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_actions = environment.action_space.n


##Hyperparamaters
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 1000
TARGET_UPDATE = 10
EDGE_CASE = 0.1


##Q-Network
class DQN(nn.Module):
    
    def __init__(self, h, w, outputs):
        super(DQN,self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 5, stride = 2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 5, stride = 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 5, stride = 2)
        self.bn3 = nn.BatchNorm2d(64)
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 64
        self.head = nn.Linear(linear_input_size, outputs)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


##Creating Transition class
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'terminal'))


##Calculating TD-error
def update_Q_target_list(memory,transition):
    Q_target_list = []
    for next_transition in memory:
        if next_transition.state == transition.next_state:
            Q_target_list.append(target_net(next_transition.state))
    if Q_target_list == []:
        return [1]
    return Q_target_list

def Q_target_next_max(memory,transition):
    Q_target_list = update_Q_target_list(memory,transition)
    return max(Q_target_list)

def Q_policy(transition):
    return policy_net(transition.state)
    
    
def TD_error(transition):
    r = transition.reward
    if transition.terminal == True:
        R_target = r
    else:
        R_target = r + GAMMA * Q_target_next_max()
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
            self.error.append(None)
        self.memory[self.position] = Transition(*args)
        self.error[self.position] = TD_error((Transition(*args)))
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


##Choosing an action
steps_done = 0

def select_action(state):
    global steps_done
    not_greedy = random.random()
    eps = EPS_END + (EPS_START - EPS_END) * math.exp(-steps_done / EPS_DECAY)
    if not_greedy > eps:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
































            