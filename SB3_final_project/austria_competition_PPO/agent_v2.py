"""Competition agent for the PPO_v2 models.

The preprocessing is identical to the training pipeline in PPO_v2.py:
    camera (3, 128, 128) -> channel last -> gray scale -> resize 84 (INTER_AREA) -> stack of 8 frames.

Usage with the competition server (server.py running on --port 5000):
    python agent_v2.py --model models_v2/bc_dagger_iter8.zip --url http://localhost:5000
    (--model a.zip,b.zip,c.zip averages the actions of several models)
"""
import argparse

import cv2
import numpy as np
from stable_baselines3 import PPO


class PPOAgent:
    def __init__(self, model_path: str, n_stack: int = 8, size: int = 84, motor_scale: float = 1.0):
        # Several models may be given as a comma separated list; their actions are averaged (ensemble).
        self.models = [PPO.load(path, device='cpu') for path in model_path.split(',')]
        self.model = self.models[0]
        self.n_stack = n_stack
        self.motor_scale = motor_scale  # < 1 makes the car slower and safer
        self.size = size
        self.stack = np.zeros((size, size, n_stack), dtype=np.uint8)
        self.first = True

    def reset(self):
        self.stack[:] = 0
        self.first = True

    def _preprocess(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs).astype(np.uint8)
        if obs.shape[0] == 3:  # (3, 128, 128) -> (128, 128, 3)
            obs = np.transpose(obs, (1, 2, 0))
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return small.astype(np.uint8)

    def act(self, obs: np.ndarray) -> np.ndarray:
        frame = self._preprocess(obs)
        # Same behaviour as VecFrameStack: shift left, newest frame in the last channel,
        # older channels are zero after a reset.
        self.stack = np.roll(self.stack, shift=-1, axis=-1)
        self.stack[..., -1] = frame
        actions = [np.asarray(m.predict(self.stack, deterministic=True)[0], dtype=np.float32).reshape(-1)[:2]
                   for m in self.models]
        action = np.mean(actions, axis=0).astype(np.float32)
        action[0] *= self.motor_scale
        return action


if __name__ == '__main__':
    from client import connect

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path of the trained PPO model (.zip).')
    parser.add_argument('--url', type=str, default='http://localhost:5000', help='The url of the server.')
    args = parser.parse_args()

    agent = PPOAgent(args.model)
    connect(agent, url=args.url)
