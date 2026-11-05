"""Score a model like the competition server does (without the HTTP layer), using the staff RaceEnv.

Scenario 'austria_competition', collisions respawn the car and shorten the time limit,
score = lap + progress - 1.

Usage: PYTHONPATH=. python evaluate_competition.py <model.zip> [episodes]
"""
import sys

import numpy as np

from agent_v2 import PPOAgent
from racecar_gym.env_competition import RaceEnv  # the staff environment, not the training wrapper


def run_episode(agent: PPOAgent, max_steps: int = 20000):
    env = RaceEnv(scenario='austria_competition', render_mode='rgb_array_birds_eye', reset_when_collision=True)
    obs, info = env.reset()
    agent.reset()
    terminal, steps = False, 0
    while not terminal and steps < max_steps:
        obs, _, terminal, _, info = env.step(agent.act(obs))
        steps += 1
    score = int(info['lap']) + info['progress'] - 1.0
    return score, info.get('n_collision', 0), info['time'], steps


if __name__ == '__main__':
    agent = PPOAgent(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    scores = []
    for ep in range(n):
        score, crashes, t, steps = run_episode(agent)
        scores.append(score)
        print(f'episode {ep}: score {score:.3f}  collisions {crashes}  env_time {t:.1f}s  steps {steps}', flush=True)
    print(f'SCORE mean {np.mean(scores):.3f}  min {np.min(scores):.3f}  max {np.max(scores):.3f}')
