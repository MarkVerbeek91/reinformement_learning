class Parameters:
    GAMMA = 0.95
    BATCH_SIZE = 64
    EPISODES_MAX: int = 1000


class Epsilon:
    start = 1.0
    min_value = 0.001
    decay = 0.999


class EnvironmentSettings:
    max_steps = 500
