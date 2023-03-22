from keras.layers import Dense, Input
from keras.models import Model, load_model
from keras.optimizers import RMSprop


def create_network_model(input_shape, action_space):
    input_nodes = Input(input_shape)

    # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(4) and Hidden Layer with 512 nodes
    network_nodes = Dense(
        512,
        input_shape=input_shape,
        activation="relu",
        kernel_initializer="he_uniform",
    )(input_nodes)

    # Hidden layer with 256 nodes
    network_nodes = Dense(
        256,
        activation="relu",
        kernel_initializer="he_uniform",
    )(network_nodes)

    # Hidden layer with 64 nodes
    network_nodes = Dense(
        64,
        activation="relu",
        kernel_initializer="he_uniform",
    )(network_nodes)

    # Output Layer with # of actions: 2 nodes (left, right)
    network_nodes = Dense(
        action_space,
        activation="linear",
        kernel_initializer="he_uniform",
    )(network_nodes)

    model = Model(
        inputs=input_nodes,
        outputs=network_nodes,
        name="CartPole_DQN_model",
    )
    model.compile(
        loss="mse",
        optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01),
        metrics=["accuracy"],
    )

    model.summary()
    return model


def load_model_from_file(file_path: str):
    return load_model(file_path)
