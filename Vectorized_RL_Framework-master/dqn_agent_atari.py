import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from base_agent import DQNBaseAgent
from models.atari_model import AtariNetDQN, AtariNetDuelingDQN
import envpool
import random

class AtariDQNAgent(DQNBaseAgent):
	def __init__(self, config):
		super(AtariDQNAgent, self).__init__(config)
		
		self.env = envpool.make(config["env_id"], env_type="gym", num_envs=self.num_envs, episodic_life=True, reward_clip=True)
		self.test_env = envpool.make(config["env_id"], env_type="gym", num_envs=self.eval_episode, episodic_life=False, reward_clip=False)
		# self.behavior_net = AtariNetDQN(self.env.action_space.n)
		self.behavior_net = AtariNetDuelingDQN(self.env.action_space.n)
		self.behavior_net.to(self.device)
		# self.target_net = AtariNetDQN(self.env.action_space.n)
		self.target_net = AtariNetDuelingDQN(self.env.action_space.n)
		self.target_net.to(self.device)
		self.target_net.load_state_dict(self.behavior_net.state_dict())
		self.target_net.eval()
		self.lr = config["learning_rate"]
		self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4)
		
	def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
		observation = torch.from_numpy(observation)
		observation = observation.to(self.device, dtype=torch.float32)
		if random.random() < epsilon:
			action = np.random.randint(0, action_space, size=observation.shape[0])
		else:
			action = self.behavior_net(observation).argmax(dim=1).cpu().numpy()
			
		return action
	
	def evaluate(self):
		print("==============================================")
		print("Evaluating...")
		self.behavior_net.eval()
		episode_rewards = [0] * self.eval_episode
		all_rewards = [0] * self.eval_episode
		zero_reward_counter = [0] * self.eval_episode
		all_done = [False] * self.eval_episode
		observations, infos = self.test_env.reset()
		while True:
			actions = self.decide_agent_actions(observations, self.eval_epsilon, self.test_env.action_space.n)
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
		self.behavior_net.train()
		return avg

if __name__ == '__main__':

	config = {
		"gpu": True,
		"training_steps": 1e8,
		"gamma": 0.99,
		"batch_size": 32,
		"eps_min": 0.1,
		"warmup_steps": 20000,
		"eps_decay": 1000000,
		"eval_epsilon": 0.01,
		"use_double": False,
		"replay_buffer_capacity": 100000,
		"logdir": 'log/dqn_eval/',
		"update_freq": 4,
		"update_target_freq": 10000,
		"learning_rate": 0.0000625,
		"env_id": 'Enduro-v5',
		"num_envs": 4,
	}
	agent = AtariDQNAgent(config)
	agent.load_and_evaluate("model_weights/Enduro/model_19333120_1384.pth")
