import gymnasium as gym

from reinforcement_learning.agents.dqn import DQNAgent

if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="human")

    learning_agent = DQNAgent(env=env)

    learning_agent.load("cartpole-dqn.h5")

    learning_agent.test()
