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
from algos.agents import A2CAgent
from algos.models import ActorCnn, CriticCnn
from algos.preprocessing.stack_frame import preprocess_frame, stack_frame

env = gym.make('ALE/Pong-v5', render_mode="rgb_array")
env.seed(0)

# %%
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# %%
def stack_frames(frames, state, is_new=False):
    frame = preprocess_frame(state, (30, -4, -12, 4), 84)
    frames = stack_frame(frames, frame, is_new)

    return frames
    
INPUT_SHAPE = (4, 84, 84)
ACTION_SIZE = env.action_space.n
SEED = 0
GAMMA = 0.99           # discount factor
ALPHA= 0.00001          # Actor learning rate
BETA = 0.00005          # Critic learning rate
UPDATE_EVERY = 10      # how often to update the network 

agent = A2CAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, GAMMA, ALPHA, BETA, UPDATE_EVERY, ActorCnn, CriticCnn)

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
            action, log_prob, entropy = agent.act(state)
            observation, reward, terminated, truncated, info= env.step(action)
            score += reward
            next_state = stack_frames(state, observation, False)
            agent.step(state, log_prob, entropy, reward, terminated, next_state)
            state = next_state
            if terminated:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        
        if i_episode % 100 == 0:
            torch.save(agent.actor_net.state_dict(), f'models/actor_model_a2c_{i_episode}.pth')
            torch.save(agent.critic_net.state_dict(), f'models/critic_model_a2c_{i_episode}.pth')
        clear_output(True)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
    
    torch.save(agent.actor_net.state_dict(), 'models/actor_model_a2c.pth')
    torch.save(agent.critic_net.state_dict(), 'models/critic_model_a2c.pth')
    return scores

# %%
start_epoch = 4200
agent.actor_net.load_state_dict(torch.load(f'models/actor_model_a2c_{start_epoch}.pth'))
agent.critic_net.load_state_dict(torch.load(f'models/critic_model_a2c_{start_epoch}.pth'))
scores = train(start_epoch + 10000)

# %%
import imageio
env.reset()
score = 0
state = stack_frames(None, env.render(), True)
to_animate = []	
while True:
    to_animate.append(env.render())
    action, log_prob, entropy = agent.act(state)
    observation, reward, terminated, truncated, info= env.step(action)
    score += reward
    next_state = stack_frames(state, observation, False)
    state = next_state
    if terminated:
        break
env.close()
imageio.mimsave('gifs/a2c.gif', to_animate)

# %%



