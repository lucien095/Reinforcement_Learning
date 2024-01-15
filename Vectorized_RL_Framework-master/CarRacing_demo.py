from ppo_agent_atari import AtariPPOAgent
from dqn_agent_atari import AtariDQNAgent
from td3_agent_CarRacing import CarRacingTD3Agent
from EnvironmentWrapper.CarRacingEnv import CarRacingEnvironment

if __name__ == '__main__':
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
		"logdir": 'log/CarRacing/td3_demo/',
		"update_freq": 2,
	}
    agent1 = CarRacingTD3Agent(config)
    agent1.load("model_weights/CarRacing/model_1367688_869.pth")
    agent2 = CarRacingTD3Agent(config)
    agent2.load("model_weights/CarRacing/model_463024_786.pth")

    env = CarRacingEnvironment(test=True)

    all_rewards = []
    eval_episode = 10
    for episode in range(eval_episode):
        total_reward = 0
        state, infos = env.reset()
        for t in range(10000):
            env.render()
            action1 = agent1.decide_agent_actions(state, 0.0, 0.035)
            action2 = agent2.decide_agent_actions(state, 0.0, 0.035)
            action = (action1 + action2) / 2
            next_state, reward, terminates, truncates, _ = env.step(action)
            # print(f"action: {action}")
            total_reward += reward
            state = next_state
            if terminates or truncates:
                print(
                    'Episode: {}\tLength: {:3d}\tTotal reward: {:.2f}'
                    .format(episode+1, t, total_reward))
                all_rewards.append(total_reward)
                break

    avg = sum(all_rewards) / eval_episode
    print(f"average score: {avg}")
    



