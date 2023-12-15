import numpy as np
import random
import gymnasium as gym
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.optimizers import Adam
import methods.ppf as ppf
import tqdm

class DQLAgent:
    def __init__(self, env, lr=0.001, gamma=0.99, epsilon=1.0):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.model = self._build_model()
        self.memory = []

    def _build_model(self):
        model = Sequential()
        model.add(Input((84,84,3)))
        model.add(Conv2D(filters = 32, kernel_size = (8,8), strides = 3, data_format="channels_last", activation='relu', kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Conv2D(filters = 64, kernel_size = (4,4), strides = 2, data_format="channels_last", activation='relu', kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Conv2D(filters = 64, kernel_size = (3,3), strides = 1, data_format="channels_last", activation='relu', kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Flatten())
        model.add(Dense(512,activation = 'relu', kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Dense(self.env.action_space.n, activation = 'linear', kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
        optimizer = Adam(self.lr)
        model.compile(optimizer, loss=tf.keras.losses.Huber())
        print('\nAgent Initialized\n')
        return model

    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()  # Explore
        q_values = self.model.predict(state[np.newaxis, :, :, :])
        return np.argmax(q_values[0])  # Exploit

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        samples = random.sample(self.memory, batch_size)
        for state, action, next_state, reward, done in samples:
            q_values = self.model.predict(state[np.newaxis, :, :, :], verbose=0)
            next_q_values = self.model.predict(next_state[np.newaxis, :, :, :], verbose=0)
            if done:
                q_values[0][action] = reward  # Update the action value
            else:
                q_values[0][action] = reward + self.gamma * np.max(next_q_values)
            self.model.fit(state[np.newaxis, :, :, :], q_values, verbose=0)

    def train(self, num_episodes=10000):
        for episode in tqdm.tqdm(range(num_episodes)):
            self.env.reset()
            state = ppf.resize_frame(self.env.step(0)[0])
            done = False
            total_reward = 0
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                state = ppf.resize_frame(state)
                next_state = ppf.resize_frame(next_state)
                self.memory.append((state, action, next_state, reward, done))
                self.replay(batch_size=32)
                state = next_state
                total_reward += reward
            print(f"Episode {episode}, Total Reward: {total_reward}")
            if episode % 10 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}")