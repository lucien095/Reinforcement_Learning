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
import envpool


class AtariPPOAgent(PPOBaseAgent):
	def __init__(self, config):
		super(AtariPPOAgent, self).__init__(config)
		
		self.env = envpool.make(config["env_id"], env_type="gym", num_envs=self.num_envs, episodic_life=True, reward_clip=True)
		self.test_env = envpool.make(config["env_id"], env_type="gym", num_envs=self.eval_episode, episodic_life=False, reward_clip=False)
		self.net = AtariNet(self.env.action_space.n)
		self.net.to(self.device)
		self.lr = config["learning_rate"]
		self.update_count = config["update_ppo_epoch"]
		self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
		
	def decide_agent_actions(self, observation, eval=False):
		# add batch dimension in observation
		# get action, value, logp from net
		observation = torch.from_numpy(observation)
		observation = observation.to(self.device, dtype=torch.float32)
		if eval:
			with torch.no_grad():
				action, logp_pi, value, _ = self.net(observation, eval=True)
		else:
			action, logp_pi, value, _ = self.net(observation)

		return action, value, logp_pi
	
	def update(self):
		p_loss_counter = 0.0001
		v_loss_counter = 0.0001
		total_surrogate_loss = 0
		total_v_loss = 0
		total_entropy = 0
		total_loss = 0

		batches = self.gae_replay_buffer.extract_batch(self.discount_factor_gamma, self.discount_factor_lambda)
		sample_count = len(batches["action"])
		batch_index = np.random.permutation(sample_count)
		
		observation_batch = {}
		for key in batches["observation"]:
			observation_batch[key] = batches["observation"][key][batch_index]
		action_batch = batches["action"][batch_index]
		return_batch = batches["return"][batch_index]
		adv_batch = batches["adv"][batch_index]
		v_batch = batches["value"][batch_index]
		logp_pi_batch = batches["logp_pi"][batch_index]


		##########################
		## One Net Architecture ##
		##########################
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

				ob_train_batch = torch.from_numpy(ob_train_batch["observation_2d"])
				ob_train_batch = ob_train_batch.to(self.device, dtype=torch.float32)
				ac_train_batch = torch.from_numpy(ac_train_batch)
				ac_train_batch = ac_train_batch.to(self.device, dtype=torch.long)
				adv_train_batch = torch.from_numpy(adv_train_batch)
				adv_train_batch = adv_train_batch.to(self.device, dtype=torch.float32)
				logp_pi_train_batch = torch.from_numpy(logp_pi_train_batch)
				logp_pi_train_batch = logp_pi_train_batch.to(self.device, dtype=torch.float32)
				return_train_batch = torch.from_numpy(return_train_batch)
				return_train_batch = return_train_batch.to(self.device, dtype=torch.float32)

				_, logp, value, entropy = self.net(ob_train_batch, False, torch.squeeze(ac_train_batch))

				# policy block
				total_ratio = torch.exp(logp - logp_pi_train_batch)
				p_opt_a = total_ratio * adv_train_batch
				p_opt_b = torch.clamp(total_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_train_batch
				surrogate_loss = -torch.mean(torch.min(p_opt_a, p_opt_b))
				# value block
				value_criterion = nn.MSELoss()
				v_loss = value_criterion(value, return_train_batch)

				# loss = surrogate_loss + v_loss
				loss = surrogate_loss + self.value_coefficient * v_loss - self.entropy_coefficient * entropy
				self.optim.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(self.net.parameters(), self.max_gradient_norm)
				self.optim.step()

				total_surrogate_loss += surrogate_loss.item()
				total_v_loss += v_loss.item()
				total_entropy += entropy.item()
				total_loss += loss.item()
				p_loss_counter += 1

		self.writer.add_scalar('PPO/Loss', total_loss / p_loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Surrogate Loss', total_surrogate_loss / p_loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Value Loss', total_v_loss / p_loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Entropy', total_entropy / p_loss_counter, self.total_time_step)
		print(f"Loss: {total_loss / p_loss_counter}\
			\tSurrogate Loss: {total_surrogate_loss / p_loss_counter}\
			\tValue Loss: {total_v_loss / p_loss_counter}\
			\tEntropy: {total_entropy / p_loss_counter}\
			")
	
	def evaluate(self):
		print("==============================================")
		print("Evaluating...")
		self.net.eval()
		episode_rewards = [0] * self.eval_episode
		all_rewards = [0] * self.eval_episode
		zero_reward_counter = [0] * self.eval_episode
		all_done = [False] * self.eval_episode
		observations, infos = self.test_env.reset()
		while True:
			actions, values, logp_pis = self.decide_agent_actions(observations)
			actions = actions.cpu().detach().numpy()

			# breakout: help agent to fire
			for i in range(self.eval_episode):
				if zero_reward_counter[i] > 200:
					actions[i] = 1
					zero_reward_counter[i] = 0

			next_observations, rewards, terminates, truncates, infos = self.test_env.step(actions)
			for i in range(self.eval_episode):
				if (terminates[i] or truncates[i]) and not all_done[i]:
					print(f"env {i} terminated, reward: {episode_rewards[i]}")
					all_rewards[i] = episode_rewards[i]
					all_done[i] = True

			for i in range(self.eval_episode):
				if rewards[i] == 0:
					zero_reward_counter[i] += 1
				else:
					zero_reward_counter[i] = 0

			episode_rewards = [episode_rewards[i] + rewards[i] for i in range(self.eval_episode)]
			observations = next_observations
			
			# all episodes done, terminate
			if all(all_done):
				break
			

		avg = sum(all_rewards) / self.eval_episode
		print(f"average score: {avg}")
		print("==============================================")
		self.net.train()
		return avg


if __name__ == '__main__':

	config = {
		"gpu": True,
		"training_steps": 1e8,
		"update_sample_count": 5120,
		"discount_factor_gamma": 0.99,
		"discount_factor_lambda": 0.95,
		"clip_epsilon": 0.2,
		"max_gradient_norm": 0.5,
		"batch_size": 128,
		"logdir": 'log/pacman_eval/',
		"update_ppo_epoch": 3,
		"learning_rate": 7e-4,
		"value_coefficient": 1.0,
		"entropy_coefficient": 0.01,
		"env_id": 'MsPacman-v5',
		"num_envs": 32,
	}
	agent = AtariPPOAgent(config)
	agent.load_and_evaluate('log/pacman/model_31457280_3092.pth')





