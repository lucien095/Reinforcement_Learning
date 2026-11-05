"""Collect (camera frame -> expert action) pairs for behavior cloning / DAgger.

Every step is labeled with the privileged expert action computed from the true car state. The car itself is
driven either by the expert (with noise bursts, so the data also covers off-line states) or, in DAgger
iterations, by the current camera policy, so the data covers the states the policy really visits.

Usage (4 parallel workers):
    PYTHONPATH=. python bc_collect.py --out-dir data/iter0 --steps 30000 --workers 4
    PYTHONPATH=. python bc_collect.py --out-dir data/iter1 --steps 15000 --workers 4 --policy bc_iter0.zip --beta 0.3
"""
import argparse
import os
from multiprocessing import Pool

import cv2
import numpy as np


def preprocess(raw):
    """(3, 128, 128) camera -> (84, 84) gray, identical to the training wrappers and to agent_v2.py."""
    obs = np.transpose(np.asarray(raw).astype(np.uint8), (1, 2, 0))
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA).astype(np.uint8)


def collect(job):
    worker, out_dir, steps, policy_path, beta, hard_prob, seed = job
    import torch
    torch.set_num_threads(1)
    from racecar_gym.env_v2 import RaceEnvV2
    from expert_driver import ExpertDriver

    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    env = RaceEnvV2(hard_start_prob=hard_prob, crash_replay_prob=0.0, max_steps=1500)
    expert = ExpertDriver()
    agent = None
    if policy_path:
        from agent_v2 import PPOAgent
        agent = PPOAgent(policy_path)

    frames, actions, firsts = [], [], []
    n_episodes = 0
    while len(frames) < steps:
        obs, info = env.reset()
        n_episodes += 1
        expert_drives = agent is None or rng.random() < beta
        if agent is not None:
            agent.reset()
        burst_left, burst = 0, np.zeros(2, dtype=np.float32)
        first = True
        while True:
            frame = preprocess(obs)
            label = expert.act(info)
            if agent is not None:
                policy_action = agent.act(obs)  # also keeps the agent frame stack up to date
            if expert_drives:
                executed = label.copy()
                # Noise bursts: hold a random steering / throttle offset for a while.
                if burst_left <= 0 and rng.random() < 0.02:
                    burst_left = int(rng.integers(8, 25))
                    burst = np.array([rng.normal(0, 0.3), rng.normal(0, 0.6)], dtype=np.float32)
                if burst_left > 0:
                    executed = executed + burst
                    burst_left -= 1
                executed = executed + rng.normal(0, [0.05, 0.1]).astype(np.float32)
            else:
                executed = policy_action + rng.normal(0, [0.05, 0.1]).astype(np.float32)
            executed = np.clip(executed, -1.0, 1.0)

            frames.append(frame)
            actions.append(label)
            firsts.append(first)
            first = False

            obs, _, terminated, truncated, info = env.step(executed)
            if terminated or truncated or len(frames) >= steps:
                break

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'worker{worker}.npz')
    np.savez_compressed(path, frames=np.stack(frames), actions=np.stack(actions), first=np.array(firsts))
    return path, len(frames), n_episodes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--steps', type=int, default=30000, help='Samples per worker.')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--policy', type=str, default=None, help='Current policy (.zip) for DAgger iterations.')
    parser.add_argument('--beta', type=float, default=1.0, help='Probability that an episode is driven by the expert.')
    parser.add_argument('--hard-start-prob', type=float, default=0.6)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    jobs = [(w, args.out_dir, args.steps, args.policy, args.beta, args.hard_start_prob, args.seed * 100 + w)
            for w in range(args.workers)]
    with Pool(args.workers) as pool:
        for path, n, episodes in pool.imap_unordered(collect, jobs):
            print(f'{path}: {n} samples in {episodes} episodes', flush=True)
