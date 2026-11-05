"""Behavior cloning of the expert into the PPO CnnPolicy (same network as PPO_v2.py).

The network input is the stack of the last 8 gray frames (84x84), exactly as in PPO_v2.py. The loss is the mean
squared error between the policy mean action and the expert action. The result is a normal PPO .zip file, so it
can be evaluated with agent_v2.py / evaluate_competition.py or fine-tuned further with PPO_v2.py --resume.

Usage:
    PYTHONPATH=. python bc_train.py --data data/iter0 data/iter1 --init start.zip --out bc_iter1.zip --epochs 6
"""
import argparse
import glob
import time

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO


def load_data(dirs):
    frames, actions, firsts = [], [], []
    for d in dirs:
        for path in sorted(glob.glob(f'{d}/worker*.npz')):
            z = np.load(path)
            frames.append(z['frames'])
            actions.append(z['actions'])
            firsts.append(z['first'])
    frames = np.concatenate(frames)
    actions = np.concatenate(actions).astype(np.float32)
    firsts = np.concatenate(firsts)
    # Index of the first frame of the episode that every sample belongs to.
    start = np.zeros(len(frames), dtype=np.int64)
    current = 0
    for i, f in enumerate(firsts):
        if f:
            current = i
        start[i] = current
    return frames, actions, start


def make_batch(frames, start, idx, n_stack=8):
    """Stack of the last n_stack frames, zero filled before the episode start (like VecFrameStack)."""
    offsets = np.arange(-(n_stack - 1), 1)  # -7 ... 0
    gather = idx[:, None] + offsets[None, :]
    valid = gather >= start[idx][:, None]
    gather = np.where(valid, gather, 0)
    batch = frames[gather] * valid[:, :, None, None]  # (B, 8, 84, 84), channel first, oldest first
    return torch.from_numpy(batch.astype(np.uint8))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs='+', required=True)
    parser.add_argument('--init', required=True, help='PPO .zip whose weights are the starting point.')
    parser.add_argument('--out', required=True)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    frames, actions, start = load_data(args.data)
    n = len(frames)
    print(f'{n} samples, action std {actions.std(0).round(3)}', flush=True)

    model = PPO.load(args.init, device='cpu')
    policy = model.policy
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    weights = torch.tensor([1.0, 2.0])  # steering matters more than throttle
    t0 = time.time()
    for epoch in range(args.epochs):
        order = np.random.permutation(n)
        total, count = 0.0, 0
        for b in range(0, n - args.batch_size + 1, args.batch_size):
            idx = order[b:b + args.batch_size]
            obs = make_batch(frames, start, idx)
            target = torch.from_numpy(actions[idx])
            mean = policy.get_distribution(obs).distribution.mean
            loss = (F.mse_loss(mean, target, reduction='none') * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            count += 1
        print(f'epoch {epoch + 1}/{args.epochs} loss {total / count:.4f}  ({time.time() - t0:.0f}s)', flush=True)

    # Small exploration noise for later RL fine-tuning (log std -1.5 is about 0.22).
    with torch.no_grad():
        policy.log_std.fill_(-1.5)
    model.save(args.out)
    print('saved', args.out)
