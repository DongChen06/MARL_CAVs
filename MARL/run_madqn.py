from MADQN import MADQN
from single_agent.utils_common import agg_double_list

import sys
sys.path.append("../highway-env")
import gym
import numpy as np
import matplotlib.pyplot as plt
import highway_env

MAX_EPISODES = 20000
EPISODES_BEFORE_TRAIN = 10
EVAL_EPISODES = 3
EVAL_INTERVAL = 200

# max steps in each episode, prevent from running too long
MAX_STEPS = 100

MEMORY_CAPACITY = 1000000
BATCH_SIZE = 128
CRITIC_LOSS = "mse"
MAX_GRAD_NORM = None

REWARD_DISCOUNTED_GAMMA = 0.99
EPSILON_START = 0.99
EPSILON_END = 0.05
EPSILON_DECAY = 50000


def run():
    env = gym.make('merge-multi-agent-v0')
    env_eval = gym.make('merge-multi-agent-v0')
    state_dim = env.n_s
    action_dim = env.n_a

    madqn = MADQN(env=env, memory_capacity=MEMORY_CAPACITY,
              state_dim=state_dim, action_dim=action_dim,
              batch_size=BATCH_SIZE, max_steps=MAX_STEPS,
              reward_gamma=REWARD_DISCOUNTED_GAMMA,
              epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
              epsilon_decay=EPSILON_DECAY, max_grad_norm=MAX_GRAD_NORM,
              episodes_before_train=EPISODES_BEFORE_TRAIN)

    episodes = []
    eval_rewards = []
    while madqn.n_episodes < MAX_EPISODES:
        madqn.interact()
        if madqn.n_episodes >= EPISODES_BEFORE_TRAIN:
            madqn.train()
        if madqn.episode_done and ((madqn.n_episodes + 1) % EVAL_INTERVAL == 0):
            rewards, _ = madqn.evaluation(env_eval, EVAL_EPISODES)
            rewards_mu, rewards_std = agg_double_list(rewards)
            print("Episode %d, Average Reward %.2f" % (madqn.n_episodes + 1, rewards_mu))
            episodes.append(madqn.n_episodes + 1)
            eval_rewards.append(rewards_mu)

    episodes = np.array(episodes)
    eval_rewards = np.array(eval_rewards)

    plt.figure()
    plt.plot(episodes, eval_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend(["DQN"])
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        run(sys.argv[1])
    else:
        run()
