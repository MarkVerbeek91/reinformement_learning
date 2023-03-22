import numpy as np
import tensorflow as tf
from keras import backend as tf_tools
from keras.layers import Dense, Input
from keras.models import Model

LOSS_CLIPPING = 0.2
STANDARD_PPO_LOSS = False


def critic_ppo2_loss(values):
    def loss(y_true, y_pred):
        clipped_value_loss = values + tf_tools.clip(
            y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING
        )
        v_loss1 = (y_true - clipped_value_loss) ** 2
        v_loss2 = (y_true - y_pred) ** 2

        if STANDARD_PPO_LOSS:
            value_loss = 0.5 * tf_tools.mean(tf_tools.maximum(v_loss1, v_loss2))
        else:
            value_loss = tf_tools.mean((y_true - y_pred) ** 2)  # standard PPO loss
        return value_loss

    return loss


def norm_init(*args):
    return lambda: tf.random_normal_initializer(*args)


class CriticModel:
    def __init__(self, input_shape, lr, optimizer, verbose=False):
        input_layer = Input(input_shape)
        old_values = Input(shape=(1,))

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
        network = Dense(1, activation=None)(network)

        self.model = Model(inputs=[input_layer, old_values], outputs=network)
        self.model.compile(
            loss=[critic_ppo2_loss(old_values)], optimizer=optimizer(lr=lr)
        )

        if verbose:
            self.model.summary()

    def predict(self, state):
        return self.model.predict([state, np.zeros((state.shape[0], 1))])
