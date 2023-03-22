import numpy as np
import tensorflow as tf
from keras import backend as tf_tools
from keras.layers import Dense, Input
from keras.models import Model


class ActorModel:
    def __init__(self, input_shape, action_space, lr, optimizer, verbose=False):
        input_layer = Input(input_shape)
        self.action_space = action_space

        network = Dense(
            512,
            activation="relu",
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        )(input_layer)
        network = Dense(
            256,
            activation="relu",
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        )(network)
        network = Dense(
            64,
            activation="relu",
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        )(network)
        network = Dense(self.action_space, activation="tanh")(network)

        self.model = Model(inputs=input_layer, outputs=network)
        self.model.compile(loss=self.ppo_loss_continuous, optimizer=optimizer(lr=lr))

        if verbose:
            print(self.model.summary())

    def ppo_loss_continuous(self, y_true, y_pred):
        (
            advantages,
            actions,
            logp_old_ph,
        ) = (
            y_true[:, :1],
            y_true[:, 1 : 1 + self.action_space],
            y_true[:, 1 + self.action_space],
        )
        LOSS_CLIPPING = 0.2
        logp = self.gaussian_likelihood(actions, y_pred)

        ratio = tf_tools.exp(logp - logp_old_ph)

        p1 = ratio * advantages
        p2 = tf.where(
            advantages > 0,
            (1.0 + LOSS_CLIPPING) * advantages,
            (1.0 - LOSS_CLIPPING) * advantages,
        )  # minimum advantage

        actor_loss = -tf_tools.mean(tf_tools.minimum(p1, p2))

        return actor_loss

    def gaussian_likelihood(self, actions, prediction):  # for keras custom loss
        log_std = -0.5 * np.ones(self.action_space, dtype=np.float32)
        pre_sum = -0.5 * (
            ((actions - prediction) / (tf_tools.exp(log_std) + 1e-8)) ** 2
            + 2 * log_std
            + tf_tools.log(2 * np.pi)
        )
        return tf_tools.sum(pre_sum, axis=1)

    def predict(self, state):
        return self.model.predict(state)
