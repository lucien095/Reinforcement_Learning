import argparse
import json
import numpy as np
import requests

import gymnasium as gym

from stable_baselines3 import PPO
import time
import cv2
import numpy as np
from collections import deque
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
step_cnt = 0

model = PPO.load('log_as/rl_model_1700000_steps.zip')

frames = deque(maxlen=8)


def connect(agent, url: str = 'http://localhost:5000'):
    while True:
        global step_cnt
        # Get the observation
        response = requests.get(f'{url}')
        if json.loads(response.text).get('error'):
            print(json.loads(response.text)['error'])
            break
        obs = json.loads(response.text)['observation']
        obs = np.array(obs).astype(np.uint8)

        if step_cnt == 0:
            obs = np.dstack((obs[0], obs[1], obs[2]))
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
            obs = cv2.resize(obs, (84, 84))

            for _ in range(frames.maxlen):
                frames.append(obs)
            obs = np.stack(frames, axis=0)
        else:
            obs = np.dstack((obs[0], obs[1], obs[2]))
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
            obs = cv2.resize(obs, (84, 84))

            frames.append(obs)
            obs = np.stack(frames, axis=0)

        # Decide an action based on the observation (Replace this with your RL agent logic)

        action_to_take, _states = model.predict(obs)  # Replace with actual action
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
