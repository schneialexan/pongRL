import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from methods import DQL, DDQN, A3C

class Agent:
    def __init__(self, env, method='DQN', lr=0.001, gamma=0.99, epsilon=1.):
        self.env = env
        self.method = method
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        if method == 'DQN':
            self.agent = DQL.DQLAgent(self.env, self.lr, self.gamma, self.epsilon)
        elif method == 'DoubleDQN':
            self.agent = DDQN.DoubleDQNAgent(self.env, self.lr, self.gamma, self.epsilon)
        elif method == 'A3C':
            self.agent = A3C.A3CAgent(self.env, self.lr, self.gamma)
        else:
            raise ValueError("Unsupported method")

    def train(self, num_episodes=10000):
        self.agent.train(num_episodes)  # Call the appropriate method-specific training function

if __name__ == "__main__":
    env = gym.make('PongDeterministic-v4')

    
