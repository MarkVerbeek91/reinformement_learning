from multiprocessing import Pipe

import gymnasium as gym
import numpy as np
from keras.optimizers import Adam
from tensorboardX import SummaryWriter

from reinforcement_learning.models.actor import ActorModel
from reinforcement_learning.models.critic import CriticModel
from reinforcement_learning.models.math import gaussian_likelihood, get_gaes
from reinforcement_learning.tools.environment import Environment


class PPOAgent:
    # PPO Main Optimization Algorithm
    def __init__(self, env_name, verbose=False):
        # Initialization
        # Environment and PPO parameters
        self.verbose = verbose
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.action_size = self.env.action_space.shape[0]
        self.state_size = self.env.observation_space.shape
        self.EPISODES = 200000  # total episodes to train through all environments
        self.episode = 0  # used to track the episodes total count of episodes played through all thread environments
        self.max_average = 0  # when average score is above 0 model will be saved
        self.lr = 0.00025
        self.epochs = 10  # training epochs
        self.shuffle = True
        self.training_batch = 512
        self.optimizer = Adam

        self.replay_count = 0
        self.writer = SummaryWriter(
            comment="_"
            + self.env_name
            + "_"
            + self.optimizer.__name__
            + "_"
            + str(self.lr)
        )

        # Instantiate plot memory
        self.scores_, self.episodes_, self.average_ = (
            [],
            [],
            [],
        )  # used in matplotlib plots

        # Create Actor-Critic network models
        self.actor = ActorModel(
            input_shape=self.state_size,
            action_space=self.action_size,
            lr=self.lr,
            optimizer=self.optimizer,
            verbose=verbose,
        )
        self.critic = CriticModel(
            input_shape=self.state_size,
            lr=self.lr,
            optimizer=self.optimizer,
            verbose=verbose,
        )

        self.Actor_name = f"{self.env_name}_PPO_Actor.h5"
        self.Critic_name = f"{self.env_name}_PPO_Critic.h5"
        # self.load() # uncomment to continue training from old weights

        # do not change bellow
        self.log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
        self.std = np.exp(self.log_std)

    def act(self, state):
        # Use the network to predict the next action to take, using the model
        prediction = self.actor.predict(state)

        low, high = -1.0, 1.0  # -1 and 1 are boundaries of tanh
        action = (
            prediction + np.random.uniform(low, high, size=prediction.shape) * self.std
        )
        action = np.clip(action, low, high)

        logp_t = gaussian_likelihood(action, prediction, self.log_std)

        return action, logp_t

    def replay(self, states, actions, rewards, dones, next_states, logp_ts):
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        logp_ts = np.vstack(logp_ts)

        # Get Critic network predictions
        values = self.critic.predict(states)
        next_values = self.critic.predict(next_states)

        # Compute discounted rewards and advantages
        # discounted_r = self.discount_rewards(rewards)
        # advantages = np.vstack(discounted_r - values)
        advantages, target = get_gaes(
            rewards, dones, np.squeeze(values), np.squeeze(next_values)
        )
        """
        pylab.plot(adv,'.')
        pylab.plot(target,'-')
        ax=pylab.gca()
        ax.grid(True)
        pylab.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06)
        pylab.show()
        if str(episode)[-2:] == "00": pylab.savefig(self.env_name+"_"+self.episode+".png")
        """
        # stack everything to numpy array
        # pack all advantages, predictions and actions to y_true and when they are received
        # in custom loss function we unpack it
        y_true = np.hstack([advantages, actions, logp_ts])

        # training Actor and Critic networks
        a_loss = self.actor.model.fit(
            states, y_true, epochs=self.epochs, verbose=0, shuffle=self.shuffle
        )
        c_loss = self.critic.model.fit(
            [states, values],
            target,
            epochs=self.epochs,
            verbose=0,
            shuffle=self.shuffle,
        )

        # calculate loss parameters
        # should be done in loss, but couldn't find working way how to do that with disabled eager execution
        pred = self.actor.predict(states)
        log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
        logp = gaussian_likelihood(actions, pred, log_std)
        approx_kl = np.mean(logp_ts - logp)
        approx_ent = np.mean(-logp)

        self.writer.add_scalar(
            "Data/actor_loss_per_replay",
            np.sum(a_loss.history["loss"]),
            self.replay_count,
        )
        self.writer.add_scalar(
            "Data/critic_loss_per_replay",
            np.sum(c_loss.history["loss"]),
            self.replay_count,
        )
        self.writer.add_scalar(
            "Data/approx_kl_per_replay", approx_kl, self.replay_count
        )
        self.writer.add_scalar(
            "Data/approx_ent_per_replay", approx_ent, self.replay_count
        )
        self.replay_count += 1

    def load(self):
        self.actor.model.load_weights(self.Actor_name)
        self.critic.model.load_weights(self.Critic_name)

    def save(self):
        self.actor.model.save_weights(self.Actor_name)
        self.critic.model.save_weights(self.Critic_name)

    def run_batch(self):
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size[0]])
        done, score = False, 0
        while True:
            # Instantiate or reset games memory
            states, next_states, actions, rewards, dones, logp_ts = (
                [],
                [],
                [],
                [],
                [],
                [],
            )
            for t in range(self.training_batch):
                if self.verbose:
                    self.env.render()
                # Actor picks an action
                action, logp_t = self.act(state)
                # Retrieve new state, reward, and whether the state is terminal
                next_state, reward, done, *_ = self.env.step(action[0])
                # Memorize (state, next_states, action, reward, done, logp_ts) for training
                states.append(state)
                next_states.append(np.reshape(next_state, [1, self.state_size[0]]))
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                logp_ts.append(logp_t[0])
                # Update current state shape
                state = np.reshape(next_state, [1, self.state_size[0]])
                score += reward
                if done:
                    self.episode += 1
                    print(f"episode: {self.episode}/{self.EPISODES}, score: {score}")
                    self.writer.add_scalar(
                        f"Workers:{1}/score_per_episode", score, self.episode
                    )
                    self.writer.add_scalar(
                        f"Workers:{1}/learning_rate", self.lr, self.episode
                    )
                    # self.writer.add_scalar(f'Workers:{1}/average_score', average, self.episode)

                    state, done, score = self.env.reset(), False, 0
                    state = np.reshape(state, [1, self.state_size[0]])

            self.replay(states, actions, rewards, dones, next_states, logp_ts)

            if self.episode >= self.EPISODES:
                break

        self.env.close()

    def run_multi_processes(self, num_worker=4):
        works, parent_conns, child_conns = [], [], []
        for idx in range(num_worker):
            parent_conn, child_conn = Pipe()
            work = Environment(
                idx,
                child_conn,
                self.env_name,
                self.state_size[0],
                self.action_size,
                True,
            )
            work.start()
            works.append(work)
            parent_conns.append(parent_conn)
            child_conns.append(child_conn)

        states = [[] for _ in range(num_worker)]
        next_states = [[] for _ in range(num_worker)]
        actions = [[] for _ in range(num_worker)]
        rewards = [[] for _ in range(num_worker)]
        dones = [[] for _ in range(num_worker)]
        logp_ts = [[] for _ in range(num_worker)]
        score = [0 for _ in range(num_worker)]

        state = [0 for _ in range(num_worker)]
        for worker_id, parent_conn in enumerate(parent_conns):
            state[worker_id] = parent_conn.recv()

        while self.episode < self.EPISODES:
            # get batch of action's and log_pi's
            action, logp_pi = self.act(
                np.reshape(state, [num_worker, self.state_size[0]])
            )

            for worker_id, parent_conn in enumerate(parent_conns):
                parent_conn.send(action[worker_id])
                actions[worker_id].append(action[worker_id])
                logp_ts[worker_id].append(logp_pi[worker_id])

            for worker_id, parent_conn in enumerate(parent_conns):
                next_state, reward, done, _ = parent_conn.recv()

                states[worker_id].append(state[worker_id])
                next_states[worker_id].append(next_state)
                rewards[worker_id].append(reward)
                dones[worker_id].append(done)
                state[worker_id] = next_state
                score[worker_id] += reward

                if done:
                    self.writer.add_scalar(
                        f"Workers:{num_worker}/score_per_episode",
                        score[worker_id],
                        self.episode,
                    )
                    self.writer.add_scalar(
                        f"Workers:{num_worker}/learning_rate", self.lr, self.episode
                    )
                    # self.writer.add_scalar(f'Workers:{num_worker}/average_score', average, self.episode)
                    score[worker_id] = 0
                    if self.episode < self.EPISODES:
                        self.episode += 1

            for worker_id in range(num_worker):
                if len(states[worker_id]) >= self.training_batch:
                    self.replay(
                        states[worker_id],
                        actions[worker_id],
                        rewards[worker_id],
                        dones[worker_id],
                        next_states[worker_id],
                        logp_ts[worker_id],
                    )

                    states[worker_id] = []
                    next_states[worker_id] = []
                    actions[worker_id] = []
                    rewards[worker_id] = []
                    dones[worker_id] = []
                    logp_ts[worker_id] = []

        # terminating processes after a while loop
        for work in works:
            work.terminate()
            print("TERMINATED:", work)
            work.join()

    def test(self, test_episodes=100):  # evaluate
        self.load()
        for e in range(101):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size[0]])
            done = False
            score = 0
            while not done:
                self.env.render()
                action = self.actor.predict(state)[0]
                state, reward, done, *_ = self.env.step(action)
                state = np.reshape(state, [1, self.state_size[0]])
                score += reward
                if done:
                    print("episode: {}/{}, score: {}".format(e, test_episodes, score))

        self.env.close()
