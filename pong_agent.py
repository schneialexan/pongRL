import gymnasium as gym

env = gym.make('ALE/Pong-ram-v5', render_mode='rgb_array')
env.reset()
print(env.render())