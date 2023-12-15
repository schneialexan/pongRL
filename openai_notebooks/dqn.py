import time
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from IPython.display import clear_output
import math
import tqdm

# %%
import sys
sys.path.append('../../')
from algos.agents.dqn_agent import DQNAgent
from algos.models.dqn_cnn import DQNCnn
from algos.preprocessing.stack_frame import preprocess_frame, stack_frame

env = gym.make('PongNoFrameskip-v0', render_mode='rgb_array')
env.unwrapped.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def stack_frames(frames, state, is_new=False):
    frame = preprocess_frame(state, (30, -4, -12, 4), 84)
    frames = stack_frame(frames, frame, is_new)

    return frames

INPUT_SHAPE = (4, 84, 84)
ACTION_SIZE = env.action_space.n
SEED = 0
GAMMA = 0.99           # discount factor
BUFFER_SIZE = 100000   # replay buffer size
BATCH_SIZE = 64        # Update batch size
LR = 0.0001            # learning rate 
TAU = 1e-3             # for soft update of target parameters
UPDATE_EVERY = 1       # how often to update the network
UPDATE_TARGET = 10000  # After which thershold replay to be started 
EPS_START = 0.99       # starting value of epsilon
EPS_END = 0.01         # Ending value of epsilon
EPS_DECAY = 100         # Rate by which epsilon to be decayed

agent = DQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)

start_epoch = 0
scores = []
scores_window = deque(maxlen=20)

epsilon_by_epsiode = lambda frame_idx: EPS_END + (EPS_START - EPS_END) * math.exp(-1. * frame_idx /EPS_DECAY)

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
            next_state, reward, done, truncated, info = env.step(action)
            score += reward
            next_state = stack_frames(state, next_state, False)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
    
        if i_episode % 100 == 0:
            torch.save(agent.policy_net.state_dict(), f'models/dqn_model_{i_episode}.pth')
        clear_output(True)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
    # save the model
    torch.save(agent.policy_net.state_dict(), f'dqn_model_{LR}.pth')
    return scores

# %%
start_epoch = 700
agent.policy_net.load_state_dict(torch.load(f'models/dqn_model_{start_epoch}.pth'))
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
imageio.mimsave('gifs/dqn.gif', to_animate)

# %%



