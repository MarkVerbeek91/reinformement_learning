import gymnasium as gym

from reinforcement_learning.agents.dqn import DQNAgent

if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    learning_agent = DQNAgent(env=env)

    try:
        learning_agent.run()
    except KeyboardInterrupt:
        print("program stopped, saving untrained model")
        learning_agent.save("CartPole-v1-unfinised.h5")
