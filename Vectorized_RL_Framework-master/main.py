from ppo_agent_atari import AtariPPOAgent
from dqn_agent_atari import AtariDQNAgent
from td3_agent_CarRacing import CarRacingTD3Agent

if __name__ == '__main__':

	# config = {
	# 	"gpu": True,
	# 	"training_steps": 1e8,
	# 	"update_sample_count": 5120,
	# 	"discount_factor_gamma": 0.99,
	# 	"discount_factor_lambda": 0.95,
	# 	"clip_epsilon": 0.2,
	# 	"max_gradient_norm": 0.5,
	# 	"batch_size": 128,
	# 	"logdir": 'log/pacman/',
	# 	"update_ppo_epoch": 3,
	# 	"learning_rate": 7e-4,
	# 	"value_coefficient": 1.0,
	# 	"entropy_coefficient": 0.01,
	# 	"env_id": 'MsPacman-v5',
	# 	"num_envs": 32,
	# }
	# agent = AtariPPOAgent(config)
	# agent.train()

	# config = {
	# 	"gpu": True,
	# 	"training_steps": 1e8,
	# 	"gamma": 0.99,
	# 	"batch_size": 32,
	# 	"eps_min": 0.1,
	# 	"warmup_steps": 20000,
	# 	"eps_decay": 1000000,
	# 	"eval_epsilon": 0.05, # 0.01 works better
	# 	"use_double": True,
	# 	"replay_buffer_capacity": 100000,
	# 	"logdir": 'log/DQN/double_dqn/',
	# 	"update_freq": 4,
	# 	"update_target_freq": 10000,
	# 	"learning_rate": 0.0000625,
	# 	"env_id": 'Breakout-v5',
	# 	"num_envs": 4,
	# }
	# agent = AtariDQNAgent(config)
	# agent.train()

	config = {
		"gpu": True,
		"training_steps": 1e8,
		"gamma": 0.99,
		"tau": 0.005,
		"batch_size": 32,
		"warmup_steps": 1000,
		"total_episode": 100000,
		"lra": 4.5e-5,
		"lrc": 4.5e-5,
		"replay_buffer_capacity": 5000,
		"logdir": 'log/CarRacing/td3_final_gnoise/',
		"update_freq": 2,
	}
	agent = CarRacingTD3Agent(config)
	agent.train()


