import argparse
import json
import numpy as np
import requests
from td3_agent_CarRacing import CarRacingTD3Agent

config = {
    "gpu": True,
    "training_steps": 1e8,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 32,
    "warmup_steps": 5000,
    "total_episode": 100000,
    "lra": 4.5e-5,
    "lrc": 4.5e-5,
    "replay_buffer_capacity": 5000,
    "logdir": 'log/CarRacing/td3_test/',
    "update_freq": 2,
    "eval_interval": 10,
    "eval_episode": 10,
    "N_frames": 4
}
car_agent = CarRacingTD3Agent(config)
car_agent.load('log/CarRacing/td3_test/model_174647_694.pth')
step_cnt = 0


def connect(agent, url: str = 'http://localhost:5000'):
    while True:
        # Get the observation
        response = requests.get(f'{url}')
        if json.loads(response.text).get('error'):
            print(json.loads(response.text)['error'])
            break
        obs = json.loads(response.text)['observation']
        obs = np.array(obs).astype(np.uint8)

        # Decide an action based on the observation (Replace this with your RL agent logic)
        global step_cnt
        action_to_take = car_agent.client_evaluate(obs, step_cnt)  # Replace with actual action
        step_cnt += 1

        # Send an action and receive new observation, reward, and done status
        response = requests.post(f'{url}', json={'action': action_to_take.tolist()})
        if json.loads(response.text).get('error'):
            print(json.loads(response.text)['error'])
            break

        result = json.loads(response.text)
        terminal = result['terminal']

        if terminal:
            print('Episode finished.')
            return


if __name__ == '__main__':
    step_cnt = 0
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, default='http://localhost:5000', help='The url of the server.')
    args = parser.parse_args()


    class RandomAgent:
        def __init__(self, action_space):
            self.action_space = action_space

        def act(self, observation):
            return self.action_space.sample()


    # Initialize the RL Agent
    import gymnasium as gym

    rand_agent = RandomAgent(
        action_space=gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32))

    connect(rand_agent, url=args.url)
