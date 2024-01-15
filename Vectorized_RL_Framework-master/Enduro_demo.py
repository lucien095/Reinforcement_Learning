from ppo_agent_atari import AtariPPOAgent
from dqn_agent_atari import AtariDQNAgent
from td3_agent_CarRacing import CarRacingTD3Agent
from EnvironmentWrapper.CarRacingEnv import CarRacingEnvironment
import gym
import numpy as np

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
    agent.load("model_weights/Enduro/model_19333120_1384.pth")

    env = gym.make("ALE/Enduro-v5", render_mode="human")
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=1)
    env = gym.wrappers.FrameStack(env, num_stack=4)

    all_rewards = []
    eval_episode = 5
    for episode in range(eval_episode):
        total_reward = 0
        state, infos = env.reset()
        state = state.__array__()
        length = 0
        while True:
            length += 1
            env.render()
            action = agent.decide_agent_actions(np.expand_dims(state, axis=0), 0.01, env.action_space.n)
            action = action[0]
            next_state, reward, terminates, truncates, _ = env.step(action)
            next_state = next_state.__array__()
            total_reward += reward
            state = next_state
            if terminates or truncates:
                print(
                    'Episode: {}\tLength: {:3d}\tTotal reward: {:.2f}'
                    .format(episode+1, length, total_reward))
                all_rewards.append(total_reward)
                break

    avg = sum(all_rewards) / eval_episode
    print(f"average score: {avg}")
    



