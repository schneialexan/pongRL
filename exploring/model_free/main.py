import agent
import environment
import matplotlib.pyplot as plt

def main():
    env = environment.PongEnv('PongDeterministic-v4', render_mode='rgb_array')
    agt = agent.PongAgent(env)
    theQ = agt.train(10000)
    plt.plot(theQ)

if __name__ == '__main__':
    main()