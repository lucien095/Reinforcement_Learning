import argparse

import gymnasium
import numpy as np
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation

from racecar_gym.env_v2 import RaceEnvV2

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor


class ChannelLastObservation(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super(ChannelLastObservation, self).__init__(env)
        obs_shape = self.observation_space.shape
        new_shape = (obs_shape[1], obs_shape[2], obs_shape[0])
        self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

    def observation(self, observation):
        return np.transpose(observation, (1, 2, 0))


def make_env():
    race_env = RaceEnvV2(
        scenario='austria_competition_collisionStop',
        render_mode='rgb_array_birds_eye',
        reset_when_collision=False,
    )
    race_env = ChannelLastObservation(race_env)
    race_env = GrayScaleObservation(race_env, keep_dim=True)
    race_env = ResizeObservation(race_env, 84)
    return race_env


def linear_schedule(initial_value: float):
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return schedule


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-num', type=int, default=50)
    parser.add_argument('--total-timesteps', type=int, default=200_000_000)
    parser.add_argument('--resume', type=str, default=None, help='Path of a checkpoint (.zip) to continue from.')
    args = parser.parse_args()

    checkpoint_callback = CheckpointCallback(
        save_freq=2000,
        save_path="./logs_as_PPO_v2/",
        name_prefix="rl_model_v2",
    )

    env = SubprocVecEnv([make_env for _ in range(args.env_num)])
    env = VecFrameStack(env, 8, channels_order='last')
    env = VecMonitor(env)

    if args.resume:
        model = PPO.load(args.resume, env=env, tensorboard_log="./PPO_v2_tensorboard/")
    else:
        model = PPO("CnnPolicy", env, verbose=1,
                    n_steps=512, batch_size=1024,
                    gamma=0.995, gae_lambda=0.95,  # longer horizon: braking pays off several seconds later
                    learning_rate=linear_schedule(3e-4),
                    ent_coef=0.02, vf_coef=0.5,
                    tensorboard_log="./PPO_v2_tensorboard/")
    model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback)
    model.save("./logs_as_PPO_v2/final_model_v2")
