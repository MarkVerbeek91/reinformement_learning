import random
from collections import deque, namedtuple

import gymnasium as gym
import numpy as np
from keras.models import load_model

from reinforcement_learning.models import create_network_model
from reinforcement_learning.parameters import Epsilon, Parameters

Memory = namedtuple("Memory", "state action reward next_state done")


class DQNAgent:
    def __init__(self, **kwargs):
        self.env = kwargs.pop("env", gym.make("CartPole-v1"))
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.memory = deque(maxlen=2000)

        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate

        self.train_start = 1000

        # create main model
        self.model = create_network_model(
            input_shape=(self.state_size,),
            action_space=self.action_size,
        )

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > Epsilon.min_value:
                self.epsilon *= Epsilon.decay

    def store_memory(self, memory):
        self.memory.append(memory)
        self.decay_epsilon()

    def decay_epsilon(self):
        if len(self.memory) > self.train_start and self.epsilon > Epsilon.min_value:
            self.epsilon *= Epsilon.decay

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        # Randomly sample minibatch from the memory
        mini_batch = random.sample(
            self.memory, min(len(self.memory), Parameters.BATCH_SIZE)
        )

        state = np.zeros((Parameters.BATCH_SIZE, self.state_size))
        next_state = np.zeros((Parameters.BATCH_SIZE, self.state_size))
        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(Parameters.BATCH_SIZE):
            state[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            next_state[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        # do batch prediction to save speed
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)

        for i in range(Parameters.BATCH_SIZE):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')
                target[i][action[i]] = reward[i] + self.gamma * (
                    np.amax(target_next[i])
                )

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=Parameters.BATCH_SIZE, verbose=0)

    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name)

    def run(self):
        for episode in range(Parameters.EPISODES_MAX):
            state = self.env.reset()[0]
            state = np.reshape(state, [1, self.state_size])
            score = 0
            while True:
                # self.env.render()
                action = self.act(state)
                next_state, reward, done, *_ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done or score == self.env._max_episode_steps - 1:
                    reward = reward
                else:
                    reward = -100

                self.remember(state, action, reward, next_state, done)
                self.store_memory(Memory(state, action, reward, next_state, done))
                state = next_state
                score += 1
                if done:
                    step = f"{episode}/{Parameters.EPISODES_MAX}"
                    print(f"episode: {step}, score: {score}, e: {self.epsilon:.2}")
                    if score == 500:
                        print("Saving trained model as cartpole-dqn.h5")
                        self.save("cartpole-dqn.h5")
                    break

                self.replay()

            if score == 500:
                break

    def test(self):
        for e in range(Parameters.EPISODES_MAX):
            state = self.env.reset()[0]
            state = np.reshape(state, [1, self.state_size])
            score = 0
            while True:
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, _, done, *_ = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_size])
                score += 1
                if done:
                    print(f"episode: {e}/{Parameters.EPISODES_MAX}, score: {score}")
                    break
