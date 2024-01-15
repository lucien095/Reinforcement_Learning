import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.gae_replay_buffer import GaeSampleMemory
from replay_buffer.replay_buffer import ReplayMemory
from abc import ABC, abstractmethod

import random

class PPOBaseAgent(ABC):
	def __init__(self, config):
		self.gpu = config["gpu"]
		self.device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")
		self.total_time_step = 0
		self.training_steps = int(config["training_steps"])
		self.update_sample_count = int(config["update_sample_count"])
		self.discount_factor_gamma = config["discount_factor_gamma"]
		self.discount_factor_lambda = config["discount_factor_lambda"]
		self.clip_epsilon = config["clip_epsilon"]
		self.max_gradient_norm = config["max_gradient_norm"]
		self.batch_size = int(config["batch_size"])
		self.value_coefficient = config["value_coefficient"]
		self.entropy_coefficient = config["entropy_coefficient"]
		self.eval_interval = config["eval_interval"]
		self.eval_episode = config["eval_episode"]

		self.gae_replay_buffer = GaeSampleMemory({
			"horizon" : config["horizon"],
			"use_return_as_advantage": False,
			"agent_count": config['env_n'],
			})

		self.writer = SummaryWriter(config["logdir"])

		self.num_envs = config['env_n']

	@abstractmethod
	def decide_agent_actions(self, observation):
		# add batch dimension in observation
		# get action, value, logp from net

		return NotImplementedError

	@abstractmethod
	def update(self):
		# sample a minibatch of transitions
		batches = self.gae_replay_buffer.extract_batch(self.discount_factor_gamma, self.discount_factor_lambda)
		# calculate the loss and update the behavior network

		return NotImplementedError


	def train(self):
		self.net.train()
		observations, infos = self.env.reset()
		episode_rewards = [0] * self.num_envs
		episode_lens = [0] * self.num_envs
		
		while self.total_time_step <= self.training_steps:
			actions, values, logp_pis = self.decide_agent_actions(observations)
			actions = actions.cpu().detach().numpy()
			values = values.cpu().detach().numpy()
			logp_pis = logp_pis.cpu().detach().numpy()
			
			next_observations, rewards, terminates, truncates, infos = self.env.step(actions)

			for i in range(self.num_envs):
				# observation must be dict before storing into gae_replay_buffer
				# dimension of reward, value, logp_pi, done must be the same
				obs = {}
				obs["observation_2d"] = np.asarray(observations[i], dtype=np.float32)
				self.gae_replay_buffer.append(i, {
						"observation": obs,
						"action": np.array([actions[i]]),
						"reward": rewards[i],
						"value": values[i],
						"logp_pi": logp_pis[i],
						"done": terminates[i],
					})

				if len(self.gae_replay_buffer) >= self.update_sample_count:
					self.update()
					self.gae_replay_buffer.clear_buffer()
					break

			episode_rewards = [episode_rewards[i] + rewards[i] for i in range(self.num_envs)]
			episode_lens = [episode_lens[i] + 1 for i in range(self.num_envs)]

			for i in range(self.num_envs):
				if terminates[i] or truncates[i]:
					if i == 0:
						self.writer.add_scalar('Train/Episode Reward', episode_rewards[0], self.total_time_step)
						self.writer.add_scalar('Train/Episode Len', episode_lens[0], self.total_time_step)
					print(f"[{len(self.gae_replay_buffer)}/{self.update_sample_count}][{self.total_time_step}/{self.training_steps}]\
						\tenv {i} \
	   					\tepisode reward: {episode_rewards[i]}\
						\tepisode len: {episode_lens[i]}\
						")
					episode_rewards[i] = 0
					episode_lens[i] = 0

			observations = next_observations
			self.total_time_step += self.num_envs
			

			if self.total_time_step % self.eval_interval == 0:
				# save model checkpoint
				avg_score = self.evaluate()
				self.save(os.path.join(self.writer.log_dir, f"model_{self.total_time_step}_{int(avg_score)}.pth"))
				self.writer.add_scalar('Evaluate/Episode Reward', avg_score, self.total_time_step)

	def evaluate(self):
		print("==============================================")
		print("Evaluating...")
		all_rewards = []
		for i in range(self.eval_episode):
			observation, info = self.test_env.reset()
			total_reward = 0
			while True:
				# self.test_env.render()
				action, _, _ = self.decide_agent_actions(observation, eval=True)
				next_observation, reward, terminate, truncate, info = self.test_env.step(action)
				total_reward += reward
				if terminate or truncate:
					print(f"episode {i+1} reward: {total_reward}")
					all_rewards.append(total_reward)
					break

				observation = next_observation
			

		avg = sum(all_rewards) / self.eval_episode
		print(f"average score: {avg}")
		print("==============================================")
		return avg
	
	# save model
	def save(self, save_path):
		torch.save(self.net.state_dict(), save_path)

	# load model
	def load(self, load_path):
		self.net.load_state_dict(torch.load(load_path))

	# load model weights and evaluate
	def load_and_evaluate(self, load_path):
		self.eval_episode = 1
		self.load(load_path)
		self.evaluate()


	

