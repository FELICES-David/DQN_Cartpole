# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:19:52 2019

@author: david
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:35:51 2019

@author: david
"""

##Packages
import gym
import random
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
state_high = env.observation_space.high
state_low = env.observation_space.low
cart_position_max = state_high[0]
cart_position_min = state_low[0]
pole_angle_max = state_high[2]
pole_angle_min = state_low[2]

##Hyperparamaters
NUM_EPISODES = 1000
MEMORY_CAPACITY = 10000
BATCH_SIZE = 256
GAMMA = 0.9
EPS = 0.001
TARGET_UPDATE = 10
PLOT_UPDATE = 40
LEARNING_RATE = 0.01


##Q-Network
class DQN(nn.Module):
    
    def __init__(self, inputs, outputs):
        super(DQN,self).__init__()
        self.linear1 = torch.nn.Linear(inputs, 16)
        self.linear2 = torch.nn.Linear(16, 32)
        self.linear3 = torch.nn.Linear(32,32)
        self.linear4 = torch.nn.Linear(32, outputs)
    
    def forward(self, x):
        if x[0][0] < cart_position_max and x[0][0] > cart_position_min and x[0][2] < pole_angle_max and x[0][2] > pole_angle_min:
            x = self.linear1(x)
            nn.ReLU()
            x = self.linear2(x)
            nn.ReLU()
            x = self.linear3(x)
            nn.ReLU()
            y_pred = self.linear4(x)
            return y_pred
        else:
            return torch.tensor([[0, 0]], device=device, dtype=torch.float)

policy_net = DQN(n_state_values,n_actions).float().to(device)
target_net = DQN(n_state_values,n_actions).float().to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters(),LEARNING_RATE)

##Creating Transition class
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'terminal')) ##everything needs to be tensors


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

replay_memory = ReplayMemory(MEMORY_CAPACITY)


##Choosing an action
def select_action(state):
    not_greedy = random.random()
    if not_greedy > EPS:
        with torch.no_grad():
            return policy_net(state).argmax().item()
    else:
        return torch.tensor([random.randrange(n_actions)], device=device, dtype=torch.long).item()


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
    batch = Transition(*zip(*transition_list))
    batch_state = torch.cat(batch.state).to(device)
    batch_action = torch.cat(batch.action).to(device)
    batch_reward = torch.cat(batch.reward).to(device)
    batch_state_action_value = policy_net(batch_state).gather(1,batch_action).to(device)
    batch_next_state = torch.cat(batch.next_state).to(device)
    batch_expected_state_action_value = target_net(batch_next_state).to(device)
    batch_expected_state_action_value = batch_expected_state_action_value.argmax() * GAMMA + batch_reward
    batch_expected_state_action_value = batch_expected_state_action_value.to(device)
    loss = nn.SmoothL1Loss().cuda()
    actual_loss = loss(batch_state_action_value,batch_expected_state_action_value).to(device)
    optimizer.zero_grad()
    actual_loss.backward()
    for param in policy_net.parameters():
        param.grad.data
    optimizer.step()


for i_episode in range(NUM_EPISODES):
    state = env.reset()
    state = torch.from_numpy(np.array([state])).float().to(device)
    for t in count():
        action = select_action(state)
        next_state, reward, done, _ = env.step(action)
        action = torch.tensor([[action]], device=device)
        reward = torch.tensor([[reward]], device=device) * (1 - done)
        next_state = torch.from_numpy(np.array([next_state])).float().to(device)
        replay_memory.push(state, action, next_state, reward, done)
        state = next_state
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            if i_episode % PLOT_UPDATE == PLOT_UPDATE - 1:
                plot_durations()
            break
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    

print('Complete')
env.close()
plt.ioff()
plt.show()