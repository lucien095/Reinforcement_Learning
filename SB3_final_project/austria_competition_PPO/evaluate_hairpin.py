# Usage: PYTHONPATH=. python evaluate_hairpin.py <checkpoint.zip> <episodes>
# Starts the car before the hairpin and counts how often it gets past progress 0.58 without crashing.
import sys, numpy as np
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
sys.path.insert(0, '.')
from PPO_v2 import ChannelLastObservation
from racecar_gym.env_v2 import RaceEnvV2

def make():
    e = RaceEnvV2(hard_start_prob=1.0, crash_replay_prob=0.0, max_steps=600)
    e.corner_entries = [0.507]  # hairpin only
    e = ChannelLastObservation(e); e = GrayScaleObservation(e, keep_dim=True); return ResizeObservation(e, 84)

model = PPO.load(sys.argv[1], device='cpu')
n = int(sys.argv[2])
env = VecFrameStack(DummyVecEnv([make]), 8, channels_order='last')
ok = 0; ends = []
for ep in range(n):
    obs = env.reset(); passed = False
    while True:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, d, info = env.step(a); i = info[0]
        if 0.58 <= i['progress'] < 0.7: passed = True
        if d[0]: break
    ok += passed; ends.append(round(i['progress'], 2))
print(f"HAIRPIN pass {ok}/{n}  end_progress={ends}")
