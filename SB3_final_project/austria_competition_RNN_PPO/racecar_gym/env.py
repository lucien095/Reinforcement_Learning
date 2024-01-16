from collections import OrderedDict
import gymnasium as gym
from numpy import array, float32
# noinspection PyUnresolvedReferences
import racecar_gym.envs.gym_api
import time
import cv2
import numpy as np
from collections import deque


class RaceEnv(gym.Env):
    camera_name = 'camera_competition'
    motor_name = 'motor_competition'
    steering_name = 'steering_competition'
    """The environment wrapper for RaceCarGym.

    - scenario: str, the name of the scenario.
        'austria_competition' or
        'plechaty_competition'

    Notes
    -----
    - Assume there are only two actions: motor and steering.
    - Assume the observation is the camera value.
    """

    def __init__(self,
                 scenario: str,
                 render_mode: str = 'rgb_array_birds_eye',
                 reset_when_collision: bool = False,
                 **kwargs):
        self.scenario = scenario.upper()[0] + scenario.lower()[1:]
        self.env_id = f'SingleAgent{self.scenario}-v0'
        self.env = gym.make(id=self.env_id,
                            render_mode=render_mode,
                            reset_when_collision=reset_when_collision,
                            **kwargs)
        self.render_mode = render_mode
        # Assume actions only include: motor and steering
        self.action_space = gym.spaces.box.Box(low=-1., high=1., shape=(2,), dtype=float32)
        # Assume observation is the camera value
        # noinspection PyUnresolvedReferences
        observation_spaces = {k: v for k, v in self.env.observation_space.items()}
        assert self.camera_name in observation_spaces, f'One of the sensors must be {self.camera_name}. Check the scenario file.'
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 128, 128), dtype=np.uint8)

        self.time_limit = 100.0
        self.cur_step = 0

        self.old_progress = 0
        self.old_steer = 0
        self.start = 0
        self.end = 0
        self.current_checkpoint = 0

    def observation_postprocess(self, obs):
        obs = obs[self.camera_name].astype(np.uint8).transpose(2, 0, 1)
        return obs

    def reset(self, *args, **kwargs: dict):
        if kwargs.get('options'):
            kwargs['options']['mode'] = 'random'
        else:
            kwargs['options'] = {'mode': 'random'}

        obs, info = self.env.reset(*args, **kwargs)
        while info["wrong_way"] or info["wall_collision"]:
            obs, info = self.env.reset(*args, **kwargs)
        obs = self.observation_postprocess(obs)
        self.start = time.process_time()

        self.cur_step = 0
        self.old_progress = info["progress"]
        self.old_steer = 0
        self.current_checkpoint = info["checkpoint"]
        return obs, info

    def step(self, actions):
        self.cur_step += 1
        motor_action, steering_action = actions

        # Add a small noise and clip the actions
        motor_scale = 0.001
        steering_scale = 0.01
        motor_action = np.clip(motor_action + np.random.normal(scale=motor_scale), -1., 1.)
        steering_action = np.clip(steering_action + np.random.normal(scale=steering_scale), -1., 1.)
        dict_actions = OrderedDict([(self.motor_name, array(motor_action, dtype=float32)),
                                    (self.steering_name, array(steering_action, dtype=float32))])

        obs, reward, terminates, truncates, infos = self.env.step(dict_actions)
        obs = self.observation_postprocess(obs)

        self.end = time.process_time()
        if self.cur_step == 1:
            self.old_steer = dict_actions['steering_competition']
        if self.end - self.start >= self.time_limit:
            truncates = True

        r_motor = 0
        p_steer = 0
        r_progress = 0
        r_track_point = 0
        p_wall_collision = 0

        r1 = 1
        r2 = -0.01
        r_motor = r1 * (dict_actions['motor_competition'] if dict_actions['motor_competition'] > 0 else -1)
        p_steer = r2 * abs(dict_actions['steering_competition'] - self.old_steer)

        my_progress = infos["progress"] + 1 if infos["progress"] - self.old_progress < -0.5 else infos["progress"]
        r_progress = 1000 * (my_progress - self.old_progress)
        r_track_point = 5 if self.current_checkpoint != infos["checkpoint"] else 0

        if infos["wall_collision"]:
            terminates = True
            p_wall_collision = -10

        self.old_progress = infos["progress"]
        self.old_steer = dict_actions['steering_competition']
        self.current_checkpoint = infos["checkpoint"]

        reward = r_motor + p_steer + r_progress + r_track_point + p_wall_collision

        return obs, reward, terminates, truncates, infos

    def render(self):
        return self.env.render()
