import copy

import numpy as np


def gaussian_likelihood(action, pred, log_std):
    # https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/sac/policies.py
    pre_sum = -0.5 * (
        ((action - pred) / (np.exp(log_std) + 1e-8)) ** 2
        + 2 * log_std
        + np.log(2 * np.pi)
    )
    return np.sum(pre_sum, axis=1)


def discount_rewards(reward):  # gaes is better
    # Compute the gamma-discounted rewards over an episode
    # We apply the discount and normalize it to avoid big variability of rewards
    gamma = 0.99  # discount rate
    running_add = 0
    discounted_r = np.zeros_like(reward)
    for i in reversed(range(0, len(reward))):
        running_add = running_add * gamma + reward[i]
        discounted_r[i] = running_add

    discounted_r -= np.mean(discounted_r)  # normalizing the result
    discounted_r /= np.std(discounted_r) + 1e-8  # divide by standard deviation
    return discounted_r


def get_gaes(
    rewards, dones, values, next_values, gamma=0.99, lamda=0.90, normalize=True
):
    deltas = [
        r + gamma * (1 - d) * nv - v
        for r, d, nv, v in zip(rewards, dones, next_values, values)
    ]
    deltas = np.stack(deltas)
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(deltas) - 1)):
        gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

    target = gaes + values
    if normalize:
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
    return np.vstack(gaes), np.vstack(target)
