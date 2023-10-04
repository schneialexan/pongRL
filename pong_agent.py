import gymnasium as gym
import matplotlib.pyplot as plt

SEED = 42

env = gym.make('ALE/Pong-v5', render_mode='rgb_array')
observation, info = env.reset(seed=SEED)


# Create a figure for plotting
plt.figure()

for _ in range(1000):
    action = env.action_space.sample()  # this is where you would insert your policy
    observation, reward, terminated, truncated, info = env.step(action)

    # Render the current frame
    plt.imshow(observation)

    # You can add additional information to the plot if needed, e.g., the reward
    plt.title(f'Reward: {reward}')

    # Display the frame
    plt.pause(0.01)

    if terminated or truncated:
        observation, info = env.reset()

env.close()