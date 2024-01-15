import gymnasium
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation

from racecar_gym.env import RaceEnv
import numpy as np

import gymnasium as gym
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, CloudpickleWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=1000,
    save_path="./logs_as_PPO/",
    name_prefix="rl_model",
    save_replay_buffer=True,
    save_vecnormalize=True,
)


# Parallel environments
class ChannelLastObservation(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super(ChannelLastObservation, self).__init__(env)
        obs_shape = self.observation_space.shape
        new_shape = (obs_shape[1], obs_shape[2], obs_shape[0])
        self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

    def observation(self, observation):
        return np.transpose(observation, (1, 2, 0))


def make_env():
    race_env = RaceEnv(
        scenario='austria_competition_collisionStop',
        # e.g., 'austria_competition', 'circle_cw_competition_collisionStop'
        render_mode='rgb_array_birds_eye',
        reset_when_collision=False,  # Only work for 'austria_competition' and 'austria_competition_collisionStop'
    )
    race_env = ChannelLastObservation(race_env)
    race_env = GrayScaleObservation(race_env, keep_dim=True)
    race_env = ResizeObservation(race_env, 84)

    return race_env


env_num = 50
if __name__ == '__main__':
    env = SubprocVecEnv([lambda: make_env() for _ in range(env_num)])
    env = VecFrameStack(env, 8, channels_order='last')
    env = VecMonitor(env)
    model = PPO("CnnPolicy", env, verbose=1, n_steps=1024, batch_size=64, tensorboard_log="./PPO_cartpole_tensorboard/")
    model.learn(total_timesteps=100000000, callback=checkpoint_callback)
