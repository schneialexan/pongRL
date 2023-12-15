import gymnasium as gym
import numpy as np

class PongEnv(gym.Env):
    def __init__(self, name, render_mode='rgb_array') -> None:
        super(PongEnv, self).__init__()
        self.env = gym.make(name, render_mode=render_mode)
        self.total_timesteps = 0
        self.memory = []
        self.action_space = self.env.action_space
    
    def take_step(self, agent, action):
        self.total_timesteps += 1
            
        next_frame, next_frames_reward, next_frame_terminal, truncated, info = self.env.step(action)
        
        # return next_state, reward, done, _ 
        return next_frame, next_frames_reward, next_frame_terminal, truncated
    
    def reset(self) -> np.ndarray:
        self.env.reset()
    
    def _get_obs(self) -> np.ndarray:
        return self.env._get_obs()