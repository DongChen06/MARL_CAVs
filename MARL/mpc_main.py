from __future__ import print_function, division
from common.utils import agg_double_list
from datetime import datetime

import argparse
import configparser
import sys

sys.path.append("../highway-env")

import gym
import os
import highway_env
import numpy as np
import matplotlib.pyplot as plt
from common.utils import VideoRecorder, copy_file, init_dir


def parse_args():
    """
    MPC for on-ramp merging
    """
    default_base_dir = "./results/"
    default_config_dir = 'configs/configs.ini'
    parser = argparse.ArgumentParser(description=('Train or evaluate policy on the environment '
                                                  'using MPC'))
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_base_dir, help="experiment base dir")
    parser.add_argument('--option', type=str, required=False,
                        default='evaluate', help="train or evaluate")
    parser.add_argument('--config-dir', type=str, required=False,
                        default=default_config_dir, help="experiment config path")
    parser.add_argument('--seed', type=int, required=False,
                        default=0, help="random seed: 0, 2000, 2021")
    parser.add_argument('--traffic_density', type=int, required=False,
                        default=1, help="1: easy,  2: medium,  3: hard")
    parser.add_argument('--safety_guarantee', type=bool, default=False,
                        help="use safety_guarantee or not")
    parser.add_argument('--n_step', type=int, required=False,
                        default=7, help="n_step: 5, 6, 7")
    parser.add_argument('--evaluation-seeds', type=str, required=False,
                        default=','.join([str(i) for i in range(0, 600, 20)]),
                        help="random seeds for evaluation, split by ,")
    args = parser.parse_args()
    return args


def main_mpc(args):
    base_dir = args.base_dir
    config_dir = args.config_dir
    config = configparser.ConfigParser()
    config.read(config_dir)

    # create an experiment folder
    now = datetime.now().strftime("%b-%d_%H_%M_%S")
    output_dir = base_dir + now
    dirs = init_dir(output_dir)

    video_dir = dirs['eval_videos']
    eval_logs = dirs['eval_logs']

    # init env
    env = gym.make('merge-multi-agent-v0')
    env.config['simulation_frequency'] = config.getint('ENV_CONFIG', 'simulation_frequency')
    env.config['duration'] = config.getint('ENV_CONFIG', 'duration')
    env.config['policy_frequency'] = config.getint('ENV_CONFIG', 'policy_frequency')
    env.config['COLLISION_REWARD'] = config.getint('ENV_CONFIG', 'COLLISION_REWARD')
    env.config['HIGH_SPEED_REWARD'] = config.getint('ENV_CONFIG', 'HIGH_SPEED_REWARD')
    env.config['HEADWAY_COST'] = config.getint('ENV_CONFIG', 'HEADWAY_COST')
    env.config['HEADWAY_TIME'] = config.getfloat('ENV_CONFIG', 'HEADWAY_TIME')
    env.config['MERGING_LANE_COST'] = config.getint('ENV_CONFIG', 'MERGING_LANE_COST')
    env.config['traffic_density'] = config.getint('ENV_CONFIG', 'traffic_density')
    env.config['safety_guarantee'] = config.getboolean('ENV_CONFIG', 'safety_guarantee')
    env.config['action_masking'] = config.getboolean('ENV_CONFIG', 'action_masking')

    test_seeds = args.evaluation_seeds
    seeds = [int(s) for s in test_seeds.split(',')]

    rewards, (vehicle_speed, vehicle_positionx, vehicle_positiony, actions, new_actions), \
                                                steps, avg_speeds, state = evaluation(env, video_dir, len(seeds))
    rewards_mu, rewards_std = agg_double_list(rewards)
    success_rate = sum(np.array(steps) == 100) / len(steps)
    avg_speeds_mu, avg_speeds_std = agg_double_list(avg_speeds)

    print("Evaluation Reward and std %.2f, %.2f " % (rewards_mu, rewards_std))
    print("Collision Rate %.2f" % (1 - success_rate))
    print("Average Speed and std %.2f , %.2f " % (avg_speeds_mu, avg_speeds_std))

    np.save(eval_logs + '/{}'.format('eval_rewards'), np.array(rewards))
    np.save(eval_logs + '/{}'.format('eval_steps'), np.array(steps))
    np.save(eval_logs + '/{}'.format('eval_avg_speeds'), np.array(avg_speeds))
    np.save(eval_logs + '/{}'.format('vehicle_speed'), np.array(vehicle_speed))
    np.save(eval_logs + '/{}'.format('vehicle_positionx'), np.array(vehicle_positionx))
    np.save(eval_logs + '/{}'.format('vehicle_positiony'), np.array(vehicle_positiony))
    np.save(eval_logs + '/{}'.format('actions'), np.array(actions))
    np.save(eval_logs + '/{}'.format('new_actions'), np.array(new_actions))
    return state


# evaluation on the environment
def evaluation(env, output_dir, eval_episodes=1):
    rewards = []
    infos = []
    avg_speeds = []
    steps = []
    vehicle_speed = []
    actions = []
    new_actions = []
    vehicle_positionx = []
    vehicle_positiony = []
    video_recorder = None
    test_seeds = args.evaluation_seeds
    seeds = [int(s) for s in test_seeds.split(',')]

    for i in range(eval_episodes):
        if i == 10:
            print()
        avg_speed = 0
        step = 0
        rewards_i = []
        infos_i = []
        done = False
        state, _ = env.reset(is_training=False, testing_seeds=seeds[i])

        n_agents = len(env.controlled_vehicles)
        rendered_frame = env.render(mode="rgb_array")
        video_filename = os.path.join(output_dir,
                                      "testing_episode{}".format(i + 1) +
                                      '.mp4')

        # Init video recording
        if video_filename is not None:
            print("Recording video to {} ({}x{}x{}@{}fps)".format(video_filename, *rendered_frame.shape, 5))
            video_recorder = VideoRecorder(video_filename, frame_size=rendered_frame.shape, fps=5)
            video_recorder.add_frame(rendered_frame)
        else:
            video_recorder = None

        while not done:
            step += 1
            """
            Adding your functions here to provide the actions for the CAVs
            for instance: there are 5 CAVs, then the action = [1, 2, 3, 4, 0]
            
            ACTIONS = {
                0: 'LANE_LEFT',
                1: 'IDLE',
                2: 'LANE_RIGHT',
                3: 'FASTER',
                4: 'SLOWER'}
            """
            # action = mpc(state)
            if step == 1:
                action = [0] * n_agents
                state, reward, done, info = env.step(action)

            o = np.zeros((len(state[:, 0]), 1))

            for x in range(len(state[:, 0])):
                if state[x, 0] == 1:
                   if state[x, 1] < 220:
                      action[x] = 1
                   if state[x, 1]>= 220 and o[x, :] == 0:
                     # mpc(state[x,:])
                     o[x,:] = o[x,:] + 1
                else:
                    if state[x, 1] >= 220 and o[x, :] == 0:
                       # mpc(state[x,:])
                      o[x, :] = o[x, :] + 1

            action = [0] * n_agents

            state, reward, done, info = env.step(action)
            avg_speed += info["average_speed"]
            rendered_frame = env.render(mode="rgb_array")
            if video_recorder is not None:
                video_recorder.add_frame(rendered_frame)

            rewards_i.append(reward)
            infos_i.append(info)

        vehicle_speed.append(info["vehicle_speed"])
        vehicle_positionx.append(info["vehicle_positionx"])
        vehicle_positiony.append(info["vehicle_positiony"])
        actions.append(info["actions"])
        new_actions.append(info["new_actions"])
        rewards.append(rewards_i)
        infos.append(infos_i)
        steps.append(step)
        avg_speeds.append(avg_speed / step)

    if video_recorder is not None:
        video_recorder.release()
    env.close()
    return rewards, (vehicle_speed, vehicle_positionx, vehicle_positiony, actions, new_actions), steps, avg_speeds,state


if __name__ == "__main__":
    args = parse_args()
    x = main_mpc(args)

