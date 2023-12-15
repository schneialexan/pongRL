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
from algos.agents import DDQNAgent
from algos.models import DDQNCnn
from algos.preprocessing.stack_frame import preprocess_frame, stack_frame

env = gym.make('ALE/Pong-v5', render_mode='rgb_array')
env.seed(0)

# %%
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# %% [markdown]
# ## Step 3: Viewing our Enviroment

# %%
print("The size of frame is: ", env.observation_space.shape)
print("No. of Actions: ", env.action_space.n)
env.reset()
plt.figure()
plt.imshow(env.render())
plt.title('Original Frame')
plt.show()

# %% [markdown]
# ## Step 5: Stacking Frame

# %%
def stack_frames(frames, state, is_new=False):
    frame = preprocess_frame(state, (30, -4, -12, 4), 84)
    frames = stack_frame(frames, frame, is_new)

    return frames
    

# %% [markdown]
# ## Step 6: Creating our Agent

# %%
INPUT_SHAPE = (4, 84, 84)
ACTION_SIZE = env.action_space.n
SEED = 0
GAMMA = 0.99           # discount factor
BUFFER_SIZE = 100000   # replay buffer size
BATCH_SIZE = 1024      # Update batch size
LR = 0.00005             # learning rate 
TAU = .1               # for soft update of target parameters
UPDATE_EVERY = 10      # how often to update the network
UPDATE_TARGET = 10000  # After which thershold replay to be started 
EPS_START = 0.99       # starting value of epsilon
EPS_END = 0.01         # Ending value of epsilon
EPS_DECAY = 750         # Rate by which epsilon to be decayed

agent = DDQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DDQNCnn)

# %% [markdown]
# ## Step 7: Watching untrained agent play

# %%

# watch an untrained agent
state = stack_frames(None, env.render(), True) 
for j in range(200):
    env.render()
    action = agent.act(state, .9)
    observation, reward, terminated, truncated, info = env.step(action)
    state = stack_frames(state, observation, False)
    if terminated:
        break 
        
env.close()

# %% [markdown]
# ## Step 8: Loading Agent
# Uncomment line to load a pretrained agent

# %%
start_epoch = 0
scores = []
scores_window = deque(maxlen=20)

# %% [markdown]
# ## Step 9: Train the Agent with DQN

# %%
epsilon_by_epsiode = lambda frame_idx: EPS_END + (EPS_START - EPS_END) * math.exp(-1. * frame_idx /EPS_DECAY)

plt.plot([epsilon_by_epsiode(i) for i in range(10000)])

# %%
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
        eps = epsilon_by_epsiode(i_episode)
        while True:
            action = agent.act(state, eps)
            observation, reward, terminated, truncated, info = env.step(action)
            score += reward
            next_state = stack_frames(state, observation, False)
            agent.step(state, action, reward, next_state, terminated)
            state = next_state
            if terminated:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        if i_episode % 100 == 0:
            torch.save(agent.policy_net.state_dict(), f'models/ddqn_model_{i_episode}.pth')
        clear_output(True)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()
        print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(i_episode, np.mean(scores_window), eps), end="")
    torch.save(agent.policy_net.state_dict(), 'DDQN_model.pth')
    return scores

# %%
start_epoch = 3000
agent.policy_net.load_state_dict(torch.load(f'models/ddqn_model_{start_epoch}.pth'))
scores = train(start_epoch + 1000)

# %% [markdown]
# ## Step 10: Watch a Smart Agent!

# %%
import imageio
env.reset()
score = 0
state = stack_frames(None, env.render(), True)
to_animate = []	
while True:
    to_animate.append(env.render())
    env.render()
    action = agent.act(state)
    next_state, reward, done, truncated, info = env.step(action)
    score += reward
    state = stack_frames(state, next_state, False)
    if done:
        print("You Final score is:", score)
        break 
env.close()
imageio.mimsave('gifs/ddqn.gif', to_animate)

# %%
env.reset()
score = 0
state = stack_frames(None, env.render(), True)
while True:
    env.render()
    action = agent.act(state, .01)
    observation, reward, terminated, truncated, info = env.step(action)
    score += reward
    state = stack_frames(state, observation, False)
    if terminated:
        print("You Final score is:", score)
        break 
env.close()


