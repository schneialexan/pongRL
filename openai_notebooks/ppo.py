# %% [markdown]
# # Pong with Poximal Policy Optimization
# 
# ## Step 1: Import the libraries

# %%
import time
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from IPython.display import clear_output
import math

# %%
import sys
sys.path.append('../../')
from algos.agents import PPOAgent
from algos.models import ActorCnn, CriticCnn
from algos.preprocessing.stack_frame import preprocess_frame, stack_frame

# %% [markdown]
# ## Step 2: Create our environment
# 
# Initialize the environment in the code cell below.
# 

# %%
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
GAMMA = 0.99           # discount factor
ALPHA= 0.00001          # Actor learning rate
BETA = 0.00001          # Critic learning rate
TAU = 0.95
BATCH_SIZE = 32
PPO_EPOCH = 5
CLIP_PARAM = 0.2
UPDATE_EVERY = 1000     # how often to update the network 


agent = PPOAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, GAMMA, ALPHA, BETA, TAU, UPDATE_EVERY, BATCH_SIZE, PPO_EPOCH, CLIP_PARAM, ActorCnn, CriticCnn)

# %%
start_epoch = 0
scores = []
scores_window = deque(maxlen=20)

def train(n_episodes=1000):
    """
    Params
    ======
        n_episodes (int): maximum number of training episodes
    """
    for i_episode in range(start_epoch + 1, n_episodes+1):
        env.reset()
        state = stack_frames(None, env.render(), True)
        score = 0
        while True:
            action, log_prob, value = agent.act(state)
            next_state, reward, done, truncated, info = env.step(action)
            score += reward
            next_state = stack_frames(state, next_state, False)
            agent.step(state, action, value, log_prob, reward, done, next_state)
            if done:
                break
            else:
                state = next_state
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        
        if i_episode % 100 == 0:
            torch.save(agent.actor_net.state_dict(), f'models/actor_model_ppo_{i_episode}.pth')
            torch.save(agent.critic_net.state_dict(), f'models/critic_model_ppo_{i_episode}.pth')
        clear_output(True)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
    
    return scores

start_epoch = 3200
agent.actor_net.load_state_dict(torch.load(f'models/actor_model_ppo_{start_epoch}.pth'))
agent.critic_net.load_state_dict(torch.load(f'models/critic_model_ppo_{start_epoch}.pth'))
scores = train(start_epoch + 1000)

env.reset()
score = 0
state = stack_frames(None, env.render(), True)
animation_frames = []
while True:
    animation_frames.append(env.render())
    action, _, _ = agent.act(state)
    next_state, reward, done, truncated, info= env.step(action)
    score += reward
    state = stack_frames(state, next_state, False)
    if done:
        print("You Final score is:", score)
        break 
env.close()

# %%
import imageio

imageio.mimsave('gifs/ppo.gif', animation_frames)


