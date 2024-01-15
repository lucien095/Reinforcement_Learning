import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.replay_buffer import ReplayMemory
from abc import ABC, abstractmethod
import cv2

from PIL import Image as im
import os
class GaussianNoise:
    def __init__(self, dim, mu=None, std=None):
        self.mu = mu if mu else np.zeros(dim)
        self.std = np.ones(dim) * std if std else np.ones(dim) * .1

    def reset(self):
        pass

    def generate(self):
        return np.random.normal(self.mu, self.std)


class OUNoiseGenerator:
    def __init__(self, mean, std_dev, theta=0.3, dt=5e-2):
        self.theta = theta
        self.dt = dt
        self.mean = mean
        self.std_dev = std_dev

        self.x = None

        self.reset()

    def reset(self):
        self.x = np.zeros_like(self.mean.shape)

    def generate(self):
        self.x = (self.x
                  + self.theta * (self.mean - self.x) * self.dt
                  + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape))

        return self.x


class TD3BaseAgent(ABC):
    def __init__(self, config):
        self.gpu = config["gpu"]
        self.device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")
        self.total_time_step = 0
        self.training_steps = int(config["training_steps"])
        self.batch_size = int(config["batch_size"])
        self.warmup_steps = config["warmup_steps"]
        self.total_episode = config["total_episode"]
        self.eval_interval = config["eval_interval"]
        self.eval_episode = config["eval_episode"]
        self.gamma = config["gamma"]
        self.tau = config["tau"]
        self.update_freq = config["update_freq"]

        self.replay_buffer = ReplayMemory(int(config["replay_buffer_capacity"]))
        self.writer = SummaryWriter(config["logdir"])

        self.frames = deque(maxlen=config["N_frames"])

        self.do_left_flag = False
        self.do_right_flag = False
        """self.e = 0"""

    @abstractmethod
    def decide_agent_actions(self, state, sigma=0.0):
        ### TODO ###
        # based on the behavior (actor) network and exploration noise

        return NotImplementedError

    def update(self):
        # update the behavior networks
        self.update_behavior_network()
        # update the target networks
        if self.total_time_step % self.update_freq == 0:
            self.update_target_network(self.target_actor_net, self.actor_net, self.tau)
            self.update_target_network(self.target_critic_net1, self.critic_net1, self.tau)
            self.update_target_network(self.target_critic_net2, self.critic_net2, self.tau)

    @abstractmethod
    def update_behavior_network(self):
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)

        ### TODO ###
        # calculate the loss and update the behavior network

        return NotImplementedError

    @staticmethod
    def update_target_network(target_net, net, tau):
        # update target network by "soft" copying from behavior network
        for target, behavior in zip(target_net.parameters(), net.parameters()):
            target.data.copy_((1 - tau) * target.data + tau * behavior.data)

    def train(self):
        for episode in range(self.total_episode):
            total_reward = 0

            self.do_right_flag = False
            self.do_left_flag = False
            self.noise.reset()

            state, infos = self.env.reset()

            """image_path = "train_wall_img" + str(episode)
            os.mkdir(image_path)
            tmp = state[-1]
            data = im.fromarray(tmp.astype(np.uint8))
            string = "state " + str(episode) + " - 0.png"
            data.save(f"{image_path}/{string}")"""

            for t in range(10000):
                if self.total_time_step < self.warmup_steps:
                    action = self.env.action_space.sample()
                else:
                    # exploration degree
                    sigma = max(0.1 * (1 - episode / self.total_episode), 0.01)
                    action = self.decide_agent_actions(state, sigma=sigma)

                # Force turn by human
                if state[-1][42][125] == 255:
                    action[1] = 0.5
                    self.do_right_flag = True
                elif state[-1][0][95] == 153 or state[-1][0][95] == 155:
                    action[1] = -0.6
                    self.do_left_flag = True

                next_state, reward, terminates, truncates, infos = self.env.step(action)

                """tmp = next_state[-1]
                data = im.fromarray(tmp.astype(np.uint8))
                string = "state " + str(episode) + " - " + str(t + 1) + ".png"
                data.save(f"{image_path}/{string}")"""

                # Told RL not to do wall_collision
                if self.do_left_flag and next_state[-1][0][95] == 255:
                    self.replay_buffer.append(state, action, [500], next_state, [int(False)])
                    self.do_left_flag = False
                elif self.do_right_flag and next_state[-1][42][125] != 255:
                    self.replay_buffer.append(state, action, [500], next_state, [int(False)])
                    self.do_right_flag = False
                else:
                    self.replay_buffer.append(state, action, [reward], next_state, [int(terminates)])

                if self.total_time_step >= self.warmup_steps:
                    self.update()

                self.total_time_step += 1
                total_reward += reward
                state = next_state

                if terminates or truncates:
                    self.writer.add_scalar('Train/Episode Reward', total_reward, self.total_time_step)
                    print(
                        'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}'
                        .format(self.total_time_step, episode + 1, t, total_reward))
                    break

            if (episode + 1) % self.eval_interval == 0:
                # save model checkpoint
                avg_score = self.evaluate()
                self.save(os.path.join(self.writer.log_dir, f"model_{self.total_time_step}_{int(avg_score)}.pth"))
                self.writer.add_scalar('Evaluate/Episode Reward', avg_score, self.total_time_step)

                """self.e += 1"""

    def evaluate(self):
        print("==============================================")
        print("Evaluating...")
        all_rewards = []
        for episode in range(self.eval_episode):
            total_reward = 0
            state, infos = self.env.reset()

            """image_path = "test_wall_img" + str(self.e) + "_" + str(episode)
            os.mkdir(image_path)
            tmp = state[-1]
            data = im.fromarray(tmp.astype(np.uint8))
            string = "state " + str(episode) + " - 0.png"
            data.save(f"{image_path}/{string}")"""

            for t in range(10000):
                action = self.decide_agent_actions(state)
                next_state, reward, terminates, truncates, _ = self.env.test_step(action)

                """tmp = next_state[-1]
                data = im.fromarray(tmp.astype(np.uint8))
                string = "state " + str(self.e) + " - " + str(t + 1) + ".png"
                data.save(f"{image_path}/{string}")"""

                total_reward += reward
                state = next_state
                if terminates or truncates:
                    print(
                        'Episode: {}\tLength: {:3d}\tTotal reward: {:.2f}'
                        .format(episode + 1, t, total_reward))
                    all_rewards.append(total_reward)
                    break

        avg = sum(all_rewards) / self.eval_episode
        print(f"average score: {avg}")
        print("==============================================")
        return avg

    def client_evaluate(self, state, step_cnt):
        state = np.dstack((state[0], state[1], state[2]))
        # convert to grayscale
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)

        if step_cnt == 0:
            self.frames.clear()
            for _ in range(self.frames.maxlen):
                self.frames.append(state)
        else:
            self.frames.append(state)
        state = np.stack(self.frames, axis=0)

        action = self.decide_agent_actions(state)
        # Force turn by human
        if state[-1][42][125] == 255:
            action[1] = 0.5
        elif state[-1][0][95] == 153 or state[-1][0][95] == 155:
            action[1] = -0.6

        return action

    # save model
    def save(self, save_path):
        torch.save(
            {
                'actor': self.actor_net.state_dict(),
                'critic1': self.critic_net1.state_dict(),
                'critic2': self.critic_net2.state_dict(),
            }, save_path)

    # load model
    def load(self, load_path):
        checkpoint = torch.load(load_path)
        self.actor_net.load_state_dict(checkpoint['actor'])
        self.critic_net1.load_state_dict(checkpoint['critic1'])
        self.critic_net2.load_state_dict(checkpoint['critic2'])

    # load model weights and evaluate
    def load_and_evaluate(self, load_path):
        self.load(load_path)
        self.evaluate()

