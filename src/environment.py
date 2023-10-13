import gymnasium as gym
import numpy as np

class PongEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()
    
    def step(self, action: int) -> tuple:
        pass
    
    def reset(self) -> np.ndarray:
        pass
    
    def _get_obs(self) -> np.ndarray:
        pass
    
    def render(self, mode: str = 'rgb_array') -> None:
        pass
    
    def close(self) -> None:
        pass
