import gym
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import math
import tqdm
import imageio

import sys
sys.path.append('../../')
from algos.agents.dqn_agent import DQNAgent
from algos.models.dqn_cnn import DQNCnn
from algos.preprocessing.stack_frame import preprocess_frame, stack_frame

env = gym.make('PongNoFrameskip-v4', render_mode='rgb_array')
env.unwrapped.seed(0)
env.metadata['render_fps'] = 60

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

def stack_frames(frames, state, is_new=False):
    frame = preprocess_frame(state, (30, -4, -12, 4), 84)
    frames = stack_frame(frames, frame, is_new)
    return frames

def render_episode(agent, output_file, eps=1.):
    env.reset()
    state = stack_frames(None, env.render(), True)
    score = 0
    frames = []
    actions = []
    while True:
        action = agent.act(state, eps)
        next_state, reward, done, truncated, info = env.step(action)
        frames.append(env.render())
        actions.append(action)
        score += reward
        next_state = stack_frames(state, next_state, False)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
    # Save frames as a GIF
    imageio.mimsave(output_file, frames)

def train(n_episodes=1000):
    print("Training is starting:")
    for i_episode in tqdm.tqdm(range(start_epoch + 1, n_episodes+1)):
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
        if i_episode % 250 == 0:
            render_episode(agent, f'{i_episode}_episode.gif', eps)
    return scores

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
EPS_DECAY = 100        # Rate by which epsilon to be decayed

agent = DQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)
start_epoch = 0
scores = []
scores_window = deque(maxlen=20)

epsilon_by_epsiode = lambda frame_idx: EPS_END + (EPS_START - EPS_END) * math.exp(-1. * frame_idx /EPS_DECAY)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([epsilon_by_epsiode(i) for i in range(1000)])
ax.set_xlabel('Episode #')
ax.set_ylabel('Epsilon')
ax.set_title('Epsilon by Episode')
plt.savefig('epsilon_by_episode.png')

scores = train(1000)
render_episode(agent, 'final_episode.gif')

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('scores.png')
# save model
agent.save('model.pth')