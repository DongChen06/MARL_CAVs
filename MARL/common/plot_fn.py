import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../')


def smooth(x, timestamps=9):
    # last 100
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i - timestamps)
        y[i] = float(x[start:(i + 1)].sum()) / (i - start + 1)
    return y


def plot_reward():
    reward_hard_bs = np.load('/home/orange/PycharmProjects/MARL_AD_U/MARL/results/Apr-11_15:14:34/eval_rewards.npy')
    # reward_hard_bs = np.load('/home/orange/PycharmProjects/MARL_AD_U/MARL/episode_rewards.npy')
    # reward_lstm = np.load(
    #     '/home/dong/PycharmProjects/MARL_AD_U_v0/MARL/results/Mar-20_00:38:36/episode_rewards.npy')
    # reward_lstm1 = np.load(
    #     '/home/dong/PycharmProjects/MARL_AD_U_v1/MARL/results/Mar-20_03:40:28/episode_rewards.npy')
    plt.figure()
    plt.xlabel("epochs")
    plt.ylabel("Reward")
    plt.title("Epoch Reward")
    plt.plot(smooth(reward_hard_bs), label='bs')
    # plt.plot(smooth(reward_lstm), label='lstm')
    # plt.plot(smooth(reward_lstm1), label='lstm1')
    # plt.xlim([20, 20000])
    plt.ylim([0, 80])
    plt.legend(loc="lower right", ncol=3)
    plt.show()


if __name__ == '__main__':
    plot_reward()