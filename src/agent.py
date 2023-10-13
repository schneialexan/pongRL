import numpy as np
from collections import defaultdict

class PongAgent:
    def __init__(self, env):
        self.env = env

    def policy(self, state):
        pass

    def generate_episode(self):
        pass

    def update(self):
        episode = self.generate_episode()
        pass

    def train(self, num_episodes):
        for ep in range(num_episodes):
            self.update()
            print("Training episode: ", ep)
            
    def get_best_action(self, state):
        """Return the best action for a given state."""
        state_str = str(state)
        return np.argmax(self.Q[state_str])        