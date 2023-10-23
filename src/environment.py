import gymnasium as gym
import numpy as np

class PongEnv(gym.Env):
    def __init__(self, name, render_mode='rgb_array') -> None:
        super(PongEnv, self).__init__()
        self.env = gym.make(name, render_mode=render_mode)
        self.total_timesteps = 0
        self.memory = []
        self.action_space = self.env.action_space
    
    def step(self, agent):
        self.total_timesteps += 1
        if self.total_timesteps % 50000 == 0:
            agent.model.save_weights('recent_weights.hdf5')
            print('\nWeights saved!')
        next_frame = self.env.render()
        next_frame, next_frames_reward, next_frame_terminal, truncated, info = self.step(agent.get_best_action(next_frame))
    
        new_state = [self.memory[-3][0], self.memory[-2][0], self.memory[-1][0], next_frame]
        new_state = np.stack(new_state, axis=-1) / 255.0
        new_state = np.expand_dims(new_state, axis=0)

        next_action = agent.get_best_action(new_state)

        if next_frame_terminal:
            self.memory.append((next_frame, next_action, next_frames_reward, next_frame_terminal))
            return next_frames_reward, True

        self.memory.append((next_frame, next_action, next_frames_reward, next_frame_terminal))

        return next_frames_reward, False
    
    def reset(self) -> np.ndarray:
        self.env.reset()
    
    def _get_obs(self) -> np.ndarray:
        return self.env._get_obs()