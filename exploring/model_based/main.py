import agent
import matplotlib.pyplot as plt
import gymnasium as gym

def main():
    name = 'PongDeterministic-v4'
    render_mode = 'rgb_array'
    env = gym.make(name, render_mode=render_mode)
    dqn_agent = agent.Agent(env, method='DQN')
    dqn_agent.train(1000)
    '''
    ddqn_agent = agent.Agent(env, method='DoubleDQN')
    ddqn_agent.train(1000)

    a3c_agent = agent.Agent(env, method='A3C')
    a3c_agent.train(1000)'''

if __name__ == '__main__':
    main()