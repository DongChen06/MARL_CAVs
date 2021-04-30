import torch as th
from torch import nn
import configparser

config_dir = 'configs/configs_acktr.ini'
config = configparser.ConfigParser()
config.read(config_dir)
torch_seed = config.getint('MODEL_CONFIG', 'torch_seed')
th.manual_seed(torch_seed)
th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True

import numpy as np
import os, logging
from single_agent.Memory_common import OnPolicyReplayMemory
from single_agent.Model_common import ActorCriticNetwork
from single_agent.kfac import KFACOptimizer
from common.utils import index_to_one_hot, entropy, to_tensor_var, VideoRecorder


class JointACKTR:
    """
    An multi-agent learned with ACKTR
    reference: https://github.com/ChenglongChen/pytorch-DRL
    """

    def __init__(self, env, state_dim, action_dim,
                 memory_capacity=10000, max_steps=None,
                 roll_out_n_steps=10, test_seeds=0,
                 reward_gamma=0.99, reward_scale=20.,
                 actor_hidden_size=128, critic_hidden_size=128,
                 actor_output_act=nn.functional.log_softmax, critic_loss="mse",
                 actor_lr=0.0001, critic_lr=0.0001, vf_coef=0.5, vf_fisher_coef=1.0,
                 entropy_reg=0.01, max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 use_cuda=True, reward_type='global_R', traffic_density=1):

        assert traffic_density in [1, 2, 3]
        assert reward_type in ["regionalR", "global_R"]

        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.env_state, _ = self.env.reset()
        self.n_episodes = 0
        self.n_steps = 0
        self.max_steps = max_steps
        self.reward_type = reward_type
        self.reward_gamma = reward_gamma
        self.reward_scale = reward_scale
        self.traffic_density = traffic_density
        self.test_seeds = test_seeds
        self.memory = OnPolicyReplayMemory(memory_capacity)
        self.actor_hidden_size = actor_hidden_size
        self.critic_hidden_size = critic_hidden_size
        self.actor_output_act = actor_output_act
        self.critic_loss = critic_loss
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.entropy_reg = entropy_reg
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.episodes_before_train = episodes_before_train
        self.use_cuda = use_cuda and th.cuda.is_available()
        self.roll_out_n_steps = roll_out_n_steps

        self.actor_critic = ActorCriticNetwork(self.state_dim, self.action_dim,
                                               min(self.actor_hidden_size, self.critic_hidden_size),
                                               self.actor_output_act)
        self.optimizer = KFACOptimizer(self.actor_critic, lr=min(self.actor_lr, self.critic_lr))
        self.vf_coef = vf_coef
        self.vf_fisher_coef = vf_fisher_coef
        if self.use_cuda:
            self.actor_critic.cuda()

        self.episode_rewards = [0]
        self.average_speed = [0]
        self.epoch_steps = [0]

    # agent interact with the environment to collect experience
    def interact(self):
        if (self.max_steps is not None) and (self.n_steps >= self.max_steps):
            self.env_state, _ = self.env.reset()
            self.n_steps = 0
        states = []
        actions = []
        rewards = []
        done = True
        average_speed = 0

        self.n_agents = len(self.env.controlled_vehicles)
        # take n steps
        for i in range(self.roll_out_n_steps):
            states.append(self.env_state)
            action = self.exploration_action(self.env_state, self.n_agents)
            next_state, global_reward, done, info = self.env.step(tuple(action))
            actions.append([index_to_one_hot(a, self.action_dim) for a in action])
            self.episode_rewards[-1] += global_reward
            self.epoch_steps[-1] += 1
            if self.reward_type == "regionalR":
                reward = info["regional_rewards"]
            elif self.reward_type == "global_R":
                reward = [global_reward] * self.n_agents
            rewards.append(reward)
            final_state = next_state
            average_speed += info["average_speed"]
            self.env_state = next_state

            self.n_steps += 1
            if done:
                self.env_state, _ = self.env.reset()
                break

        # discount reward
        if done:
            final_value = [0.0] * self.n_agents
            self.n_episodes += 1
            self.episode_done = True
            self.episode_rewards.append(0)
            self.average_speed[-1] = average_speed / self.epoch_steps[-1]
            self.average_speed.append(0)
            self.epoch_steps.append(0)
        else:
            self.episode_done = False
            final_action = self.action(final_state)
            final_value = self.value(final_state, final_action)

        if self.reward_scale > 0:
            rewards = np.array(rewards) / self.reward_scale

        for agent_id in range(self.n_agents):
            rewards[:, agent_id] = self._discount_reward(rewards[:, agent_id], final_value[agent_id])

        rewards = rewards.tolist()
        self.memory.push(states, actions, rewards)

    # train on a roll out batch
    def train(self):
        if self.n_episodes <= self.episodes_before_train:
            pass

        batch = self.memory.sample(self.batch_size)
        states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.n_agents, self.state_dim)
        actions_var = to_tensor_var(batch.actions, self.use_cuda).view(-1, self.n_agents, self.action_dim)
        rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, self.n_agents, 1)

        for agent_id in range(self.n_agents):
            # update actor network
            action_log_probs, values = self.actor_critic(states_var[:, agent_id, :])
            entropy_loss = th.mean(entropy(th.exp(action_log_probs)))
            action_log_probs = th.sum(action_log_probs * actions_var[:, agent_id, :], 1)
            # fisher loss
            if self.optimizer.steps % self.optimizer.Ts == 0:
                self.actor_critic.zero_grad()
                pg_fisher_loss = th.mean(action_log_probs)
                values_noise = to_tensor_var(np.random.randn(values.size()[0]), self.use_cuda)
                sample_values = (values + values_noise.view(-1, 1)).detach()
                if self.critic_loss == "huber":
                    vf_fisher_loss = - nn.functional.smooth_l1_loss(values, sample_values)
                else:
                    vf_fisher_loss = - nn.MSELoss()(values, sample_values)
                joint_fisher_loss = pg_fisher_loss + self.vf_fisher_coef * vf_fisher_loss
                self.optimizer.acc_stats = True
                joint_fisher_loss.backward(retain_graph=True)
                self.optimizer.acc_stats = False
            self.optimizer.zero_grad()
            # actor loss
            advantages = rewards_var[:, agent_id, :] - values.detach()
            pg_loss = -th.mean(action_log_probs * advantages)
            actor_loss = pg_loss - entropy_loss * self.entropy_reg
            # critic loss
            target_values = rewards_var[:, agent_id, :]
            if self.critic_loss == "huber":
                critic_loss = nn.functional.smooth_l1_loss(values, target_values)
            else:
                critic_loss = nn.MSELoss()(values, target_values)
            loss = actor_loss + critic_loss
            loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

    # predict softmax action based on state
    def _softmax_action(self, state, n_agents):
        state_var = to_tensor_var([state], self.use_cuda)

        softmax_action = []
        for agent_id in range(n_agents):
            softmax_action_var = th.exp(self.actor_critic(state_var[:, agent_id, :])[0])

            if self.use_cuda:
                softmax_action.append(softmax_action_var.data.cpu().numpy()[0])
            else:
                softmax_action.append(softmax_action_var.data.numpy()[0])
        return softmax_action

    # choose an action based on state with random noise added for exploration in training
    def exploration_action(self, state, n_agents):
        softmax_actions = self._softmax_action(state, n_agents)
        actions = []
        for pi in softmax_actions:
            actions.append(np.random.choice(np.arange(len(pi)), p=pi))
        return actions

    # choose an action based on state for execution
    def action(self, state, n_agents):
        softmax_actions = self._softmax_action(state, n_agents)
        actions = []
        for pi in softmax_actions:
            actions.append(np.random.choice(np.arange(len(pi)), p=pi))
        return actions

    # evaluate value for a state-action pair
    def value(self, state, action):
        state_var = to_tensor_var([state], self.use_cuda)
        values = [0] * self.n_agents
        for agent_id in range(self.n_agents):
            value_var = self.actor_critic(state_var[:, agent_id, :])
            if self.use_cuda:
                values[agent_id] = value_var.data.cpu().numpy()[0]
            else:
                values[agent_id] = value_var.data.numpy()[0]
        return values

    # evaluation the learned agent
    def evaluation(self, env, output_dir, eval_episodes=1, is_train=True):
        rewards = []
        infos = []
        avg_speeds = []
        steps = []
        vehicle_speed = []
        vehicle_position = []
        video_recorder = None
        seeds = [int(s) for s in self.test_seeds.split(',')]

        for i in range(eval_episodes):
            avg_speed = 0
            step = 0
            rewards_i = []
            infos_i = []
            done = False
            if is_train:
                if self.traffic_density == 1:
                    state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i], num_CAV=i + 1)
                elif self.traffic_density == 2:
                    state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i], num_CAV=i + 2)
                elif self.traffic_density == 3:
                    state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i], num_CAV=i + 4)
            else:
                state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i])

            n_agents = len(env.controlled_vehicles)
            rendered_frame = env.render(mode="rgb_array")
            video_filename = os.path.join(output_dir,
                                          "testing_episode{}".format(self.n_episodes + 1) + '_{}'.format(i) +
                                          '.mp4')
            # Init video recording
            if video_filename is not None:
                print("Recording video to {} ({}x{}x{}@{}fps)".format(video_filename, *rendered_frame.shape,
                                                                      5))
                video_recorder = VideoRecorder(video_filename,
                                               frame_size=rendered_frame.shape, fps=5)
                video_recorder.add_frame(rendered_frame)
            else:
                video_recorder = None

            while not done:
                step += 1
                action = self.action(state, n_agents)
                state, reward, done, info = env.step(action)
                avg_speed += info["average_speed"]
                rendered_frame = env.render(mode="rgb_array")
                if video_recorder is not None:
                    video_recorder.add_frame(rendered_frame)

                rewards_i.append(reward)
                infos_i.append(info)

            vehicle_speed.append(info["vehicle_speed"])
            vehicle_position.append(info["vehicle_position"])
            rewards.append(rewards_i)
            infos.append(infos_i)
            steps.append(step)
            avg_speeds.append(avg_speed / step)

        if video_recorder is not None:
            video_recorder.release()
        env.close()
        return rewards, (vehicle_speed, vehicle_position), steps, avg_speeds

    # discount roll out rewards
    def _discount_reward(self, rewards, final_value):
        discounted_r = np.zeros_like(rewards)
        running_add = final_value
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.reward_gamma + rewards[t]
            discounted_r[t] = running_add
        return discounted_r

    def load(self, model_dir, global_step=None, train_mode=False):
        save_file = None
        save_step = 0
        if os.path.exists(model_dir):
            if global_step is None:
                for file in os.listdir(model_dir):
                    if file.startswith('checkpoint'):
                        tokens = file.split('.')[0].split('-')
                        if len(tokens) != 2:
                            continue
                        cur_step = int(tokens[1])
                        if cur_step > save_step:
                            save_file = file
                            save_step = cur_step
            else:
                save_file = 'checkpoint-{:d}.pt'.format(global_step)
        if save_file is not None:
            file_path = model_dir + save_file
            checkpoint = th.load(file_path)
            print('Checkpoint loaded: {}'.format(file_path))
            # logging.info('Checkpoint loaded: {}'.format(file_path))
            self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
            if train_mode:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.actor_critic.train()
            else:
                self.actor_critic.eval()
            return True
        logging.error('Can not find checkpoint for {}'.format(model_dir))
        return False

    def save(self, model_dir, global_step):
        file_path = model_dir + 'checkpoint-{:d}.pt'.format(global_step)
        th.save({'global_step': global_step,
                 'model_state_dict': self.actor_critic.state_dict(),
                 'optimizer_state_dict': self.optimizer.state_dict()},
                file_path)
