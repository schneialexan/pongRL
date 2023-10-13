import agent
import environment

def main():
    env = environment.PongEnv()
    agt = agent.PongAgent(env)
    agt.train(10000)

if __name__ == '__main__':
    main()