import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.gae_replay_buffer import GaeSampleMemory
from base_agent import PPOBaseAgent
from models.atari_model import AtariNet
import gym


class AtariPPOAgent(PPOBaseAgent):
	def __init__(self, config):
		super(AtariPPOAgent, self).__init__(config)
		### TODO ###
		# initialize env
		# set a atari game
		env = gym.make(config["env_id"], render_mode='rgb_array')
		# resize screen to 84x84
		env = gym.wrappers.ResizeObservation(env, (84, 84))
		# convert to grayscale
		env = gym.wrappers.GrayScaleObservation(env)
		# stack 4 frames 
		env = gym.wrappers.FrameStack(env, 4)
		self.env = env

		### TODO ###
		# initialize test_env
		self.test_env = env

		self.net = AtariNet(self.env.action_space.n)
		self.net.to(self.device)
		self.lr = config["learning_rate"]
		self.update_count = config["update_ppo_epoch"]
		self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
		
	def decide_agent_actions(self, observation, eval=False):
		### TODO ###
		# add batch dimension in observation
		batch_dimension_observation = torch.from_numpy(np.array(observation)).unsqueeze(0).to(self.device, dtype=torch.float32)

		# get action, value, logp from net
		if eval:
			with torch.no_grad():
				action, value, logp, _ = self.net(batch_dimension_observation, eval=True)
		else:
			action, value, logp, _ = self.net(batch_dimension_observation)

		return action, value, logp


	
	def update(self):
		loss_counter = 0.0001
		total_surrogate_loss = 0
		total_v_loss = 0
		total_entropy = 0
		total_loss = 0
		# 從buffer提取舊資料
		batches = self.gae_replay_buffer.extract_batch(self.discount_factor_gamma, self.discount_factor_lambda)
		sample_count = len(batches["action"])
		batch_index = np.random.permutation(sample_count)
		
		observation_batch = {} # old state (observatio觀察)
		for key in batches["observation"]:
			observation_batch[key] = batches["observation"][key][batch_index]
		action_batch = batches["action"][batch_index] # old action
		return_batch = batches["return"][batch_index] # old returns = advantages + values
		adv_batch = batches["adv"][batch_index] # old advantage
		v_batch = batches["value"][batch_index] # old Value
		logp_pi_batch = batches["logp_pi"][batch_index] # old log π() 

		for _ in range(self.update_count):
			for start in range(0, sample_count, self.batch_size):
				ob_train_batch = {}
				for key in observation_batch:
					ob_train_batch[key] = observation_batch[key][start:start + self.batch_size]
				ac_train_batch = action_batch[start:start + self.batch_size]
				return_train_batch = return_batch[start:start + self.batch_size]
				adv_train_batch = adv_batch[start:start + self.batch_size]
				v_train_batch = v_batch[start:start + self.batch_size]
				logp_pi_train_batch = logp_pi_batch[start:start + self.batch_size]

				ob_train_batch = torch.from_numpy(ob_train_batch["observation_2d"]) # state
				ob_train_batch = ob_train_batch.to(self.device, dtype=torch.float32)
				ac_train_batch = torch.from_numpy(ac_train_batch)					# action
				ac_train_batch = ac_train_batch.to(self.device, dtype=torch.long)
				adv_train_batch = torch.from_numpy(adv_train_batch)					# advantage
				adv_train_batch = adv_train_batch.to(self.device, dtype=torch.float32)
				logp_pi_train_batch = torch.from_numpy(logp_pi_train_batch)			# log π()
				logp_pi_train_batch = logp_pi_train_batch.to(self.device, dtype=torch.float32)
				return_train_batch = torch.from_numpy(return_train_batch)			# returns = advantages + values
				return_train_batch = return_train_batch.to(self.device, dtype=torch.float32)

				### TODO ###
				# calculate loss and update network
				action, value, logp, entropy = self.net(ob_train_batch, False, torch.squeeze(ac_train_batch))
				
				# calculate policy loss
				ratio = torch.exp(logp) / torch.exp(logp_pi_train_batch)
				# chapter 4-1 p29 𝐿𝐶𝐿𝐼𝑃=E [min(𝑟𝑡(𝜃)𝐴𝑡,𝑐𝑙𝑖𝑝(𝑟𝑡𝜃,1−𝜖,1+𝜖)𝐴𝑡]
				# compute 𝑟𝑡(𝜃)𝐴𝑡
				CPI = ratio * adv_train_batch
				# compute 𝑐𝑙𝑖𝑝(𝑟𝑡𝜃,1−𝜖,1+𝜖)𝐴𝑡
				clip = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_train_batch
				surrogate_loss = -torch.mean(torch.min(CPI, clip))

				# calculate value loss
				value_criterion = nn.MSELoss()
				v_loss = value_criterion(value, return_train_batch)
				
				# calculate total loss
				loss = surrogate_loss + self.value_coefficient * v_loss - self.entropy_coefficient * entropy

				# update network
				self.optim.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(self.net.parameters(), self.max_gradient_norm)
				self.optim.step()

				total_surrogate_loss += surrogate_loss.item()
				total_v_loss += v_loss.item()
				total_entropy += entropy.item()
				total_loss += loss.item()
				loss_counter += 1

		self.writer.add_scalar('PPO/Loss', total_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Surrogate Loss', total_surrogate_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Value Loss', total_v_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Entropy', total_entropy / loss_counter, self.total_time_step)
		print(f"Loss: {total_loss / loss_counter}\
			\tSurrogate Loss: {total_surrogate_loss / loss_counter}\
			\tValue Loss: {total_v_loss / loss_counter}\
			\tEntropy: {total_entropy / loss_counter}\
			")
	



