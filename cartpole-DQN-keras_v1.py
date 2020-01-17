# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:35:55 2019

@author: david
"""

import random
import gym
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

print(tf.test.is_gpu_available())

ENV_NAME = "CartPole-v1"
MAX_RUN = 1000
SUCCESS = 200
MEAN_LEN = 10
nb_eval = 100
nb_replay = 10

GAMMA = 0.95
LEARNING_RATE = 0.001
MEMORY_SIZE = 1000000
BATCH_SIZE = 20
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

env = gym.make(ENV_NAME)
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(12, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)



def cartpole():
    global dqn_solver
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    means = []
    durations_t = []
    #last_trial = 0
    last_mean=0
    while run < MAX_RUN :
        if last_mean >= SUCCESS:
            print('Success !')
            break
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            #env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                durations_t.append(step)
                plt.plot(durations_t)
                if len(durations_t) > MEAN_LEN:
                    last_mean = np.mean(durations_t[-MEAN_LEN:])
                    means.append(last_mean)
                    means_plot = np.concatenate((np.zeros(MEAN_LEN-1),np.array(means)))
                    plt.plot(means_plot)
                plt.show()
                #last_trial = step
                break
            dqn_solver.experience_replay()

def evaluation(number_of_tests):
    global best_test
    global all_tests
    global mean_reward
    max_reward = 0
    best_test = [0,[],[]]
    all_tests = []
    mean_reward = 0
    for i in range(number_of_tests):
        step = 0
        cumulative_reward = 0
        state = env.reset()
        current_test = [cumulative_reward, state, []]
        state = np.reshape(state, [1, observation_space])
        while True:
            step += 1
            #env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            state_next = np.reshape(state_next, [1, observation_space])
            state = state_next
            cumulative_reward += reward
            current_test[0] = cumulative_reward
            current_test[2].append(action)
            if terminal:
                all_tests.append(current_test)
                if cumulative_reward > max_reward:
                    max_reward = cumulative_reward
                    best_test = current_test
                break
    env.close()
    for i in range(number_of_tests):
        mean_reward += all_tests[i][0]/number_of_tests
    return all_tests, best_test, mean_reward

def replay_test(test):
    for k in range(nb_replay):
        env.reset()
        env.env.state = test[1]
        i=0
        while True:
            env.render()
            _,_,terminal,_ = env.step(test[2][i])
            i += 1
            if terminal:
                break
    env.close()

if __name__ == "__main__":
    cartpole()
    evaluation(nb_eval)
    print("Best result:",best_test[0])
    print("Average score in", str(nb_eval),"tests:", mean_reward)
    print('Replaying best test...')
    replay_test(best_test)