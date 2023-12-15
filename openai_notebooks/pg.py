import time
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from IPython.display import clear_output
import math

import sys
sys.path.append('../../')
from algos.agents import ReinforceAgent
from algos.models import ActorCnn
from algos.preprocessing.stack_frame import preprocess_frame, stack_frame

env = gym.make('Pong-v0', render_mode='rgb_array')
env.seed(0)

# %%
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

def stack_frames(frames, state, is_new=False):
    frame = preprocess_frame(state, (30, -4, -12, 4), 84)
    frames = stack_frame(frames, frame, is_new)

    return frames
    

INPUT_SHAPE = (4, 84, 84)
ACTION_SIZE = env.action_space.n
SEED = 0
GAMMA = 0.1        # discount factor
LR= 0.00005          # Learning rate

agent = ReinforceAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, GAMMA, LR, ActorCnn)

start_epoch = 0
scores = []
scores_window = deque(maxlen=20)

to_animate = []
def train(n_episodes=1000):
    """
    Params
    ======
        n_episodes (int): maximum number of training episodes
    """
    rewarding_episodes = []
    for i_episode in range(start_epoch + 1, n_episodes+1):
        env.reset()
        state = stack_frames(None, env.render(), True)
        score = 0
        rewards = []
        done = False
        while True:
            action, log_prob = agent.act(state)
            next_state, reward, done, truncated, info = env.step(action)
            to_animate.append(next_state)
            score += reward
            rewards.append(reward)
            next_state = stack_frames(state, next_state, False)
            agent.step(log_prob, reward, done)
            state = next_state
            if done:
                break
        agent.learn()
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        rewarding_episodes.append(rewards)
        if i_episode % 100 == 0:
            torch.save(agent.policy_net.state_dict(), f'models/pg_model_{i_episode}.pth')
        clear_output(True)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
    # save the model
    torch.save(agent.policy_net.state_dict(), f'pg_model_{LR}.pth')
    print("\nScores: ", rewarding_episodes)
    return scores

start_epoch = 3800
agent.policy_net.load_state_dict(torch.load(f'models/pg_model_{start_epoch}.pth'))
scores = train(start_epoch + 1000)

import imageio
env.reset()
score = 0
state = stack_frames(None, env.render(), True)
to_animate = []	
while True:
    to_animate.append(env.render())
    env.render()
    action, _ = agent.act(state)
    next_state, reward, done, truncated, info = env.step(action)
    score += reward
    state = stack_frames(state, next_state, False)
    if done:
        print("You Final score is:", score)
        break 
env.close()
imageio.mimsave('gifs/pg.gif', to_animate)


