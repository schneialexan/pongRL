import the_agent
import environment
import time
from collections import deque
import numpy as np
import imageio
import preprocess_frame as ppf
import cv2

name = 'PongDeterministic-v4'

agent = the_agent.Agent(possible_actions=[0,2,3],
                        starting_mem_len=50000,
                        max_mem_len=750000,
                        starting_epsilon = 1, 
                        learn_rate = .00025)
env = environment.make_env(name)

last_100_avg = [-21]
scores = deque(maxlen = 100)
max_score = -21


# testing
#agent.model.load_weights('recent_weights.hdf5')
#agent.model_target.load_weights('recent_weights.hdf5')
#agent.epsilon = 0.0


env.reset()
for i in range(151):
    timesteps = agent.total_timesteps
    timee = time.time()
    score, to_animate = environment.play_episode(env, agent, debug = False)
    scores.append(score)
    if score > max_score:
        max_score = score
    
    if i % 10 == 0:
        frame_number = agent.total_timesteps - timesteps
        imageio.mimsave(f'gifs/restart/episode_{i}_score_{score}_frames_{frame_number}.gif', to_animate)
        print(f'Episode {i} saved.')

env.reset()
score = 0
next_frame = ppf.resize_frame(env.render())
agent.memory.frames.append(next_frame)
agent.memory.frames.append(next_frame)
agent.memory.frames.append(next_frame)
to_animate = []	
while True:
    state = [agent.memory.frames[-3], agent.memory.frames[-2], agent.memory.frames[-1], next_frame]
    state = np.moveaxis(state,0,2)/255 #We have to do this to get it into keras's goofy format of [batch_size,rows,columns,channels]
    state = np.expand_dims(state,0) 
    to_animate.append(env.render())
    action = agent.get_action(state)
    observation, reward, terminated, truncated, info= env.step(action)
    score += reward
    next_frame = ppf.resize_frame(observation)
    agent.memory.frames.append(next_frame)
    if terminated:
        break
env.close()
imageio.mimsave('gifs/final.gif', to_animate)

print(f'Done!!')