import agent
import environment

def main():
    env = environment.PongEnv('PongDeterministic-v4', render_mode='rgb_array')
    agt = agent.PongAgent(env)
    agt.train(10000)

if __name__ == '__main__':
    main()