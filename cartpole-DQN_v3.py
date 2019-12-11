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


##Hyperparamaters
NUM_EPISODES = 1000
MEMORY_CAPACITY = 30000
BATCH_SIZE = 1024
GAMMA = 0.95
EPS = 0.05
TARGET_UPDATE = 10
PLOT_UPDATE = 15
LEARNING_RATE = 0.0006
SUCCESS = 100
TRIALS_MAX = 10
EVALUATIONS = 100


##Q-Network
class DQN(nn.Module):
    
    def __init__(self, inputs, outputs):
        super(DQN,self).__init__()
        self.linear1 = torch.nn.Linear(inputs, 16)
        self.linear2 = torch.nn.Linear(16, 32)
        self.linear3 = torch.nn.Linear(32, 64)
        self.linear4 = torch.nn.Linear(64, outputs)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        nn.ReLU()
        x = self.linear4(x)
        return x

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
    
    
    ##Choosing an action
    def select_action(state):
        not_greedy = random.random()
        if not_greedy > EPS:
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
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
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
        if means[-1] >= SUCCESS:
            print('Success')
            Training = 'Success'
            break
        if means[-1] < means_max * 0.8:
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
        state = torch.from_numpy(np.array([state])).float().to(device)
        cumulative_reward = 0
        current_test = [cumulative_reward, [state]] 
        for t in count():
            env.render()
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            cumulative_reward += reward
            state = torch.from_numpy(np.array([state])).float().to(device)
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