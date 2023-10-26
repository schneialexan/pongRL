import numpy as np
from collections import defaultdict
import time

class PongAgent:
    def __init__(self, env, gamma=0.9, alpha=0.1, epsilon=1.):
        self.env = env
        self.gamma = gamma  # Discount factor for future rewards
        self.alpha = alpha  # Learning rate
        self.epsilon = epsilon  # Epsilon-greedy exploration parameter
        self.Q = defaultdict(lambda: np.zeros(self.env.action_space.n))

    def policy(self, state):
        # Epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Explore by selecting a random action
        else:
            state_str = str(state)
            return np.argmax(self.Q[state_str])  # Exploit by selecting the best action

    def generate_episode(self):
        episode = []
        state = self.env.reset()
        done = False

        while not done:
            action = self.policy(state)
            next_state, reward, done, _ = self.env.take_step(self, action)
            episode.append((state, action, reward))
            state = next_state

        return episode

    def update(self):
        episode = self.generate_episode()
        start_time = time.time()
        
        if self.env.total_timesteps % 50000 == 0:
            self.model.save_weights('recent_weights.hdf5')
            print('\nWeights saved!')
        # Q-learning update for each state-action pair in this episode
        G = 0
        score = 0
        for state, action, reward in reversed(episode):
            state_str = str(state)
            score += reward
            G = self.gamma * G + reward  # Calculate the cumulative reward
            self.epsilon = max(self.epsilon * 0.9999999, 0.1)
            # Q-learning update
            self.Q[state_str][action] += self.alpha * (G - self.Q[state_str][action])
        # print episode stats
        print(f'Duration: {time.time() - start_time}')
        print(f'Score: {score}')
        print(f'Epsilon: {self.epsilon}\n')


    def train(self, num_episodes):
        for ep in range(num_episodes):
            print(f'Episode {ep + 1}/{num_episodes}')
            self.update()
        # Return the learned policy
        return self.Q
            

    def get_best_action(self, state):
        state_str = str(state)
        return np.argmax(self.Q[state_str])
