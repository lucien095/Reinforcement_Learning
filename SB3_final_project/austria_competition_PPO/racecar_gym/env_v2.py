import random
import time
from collections import OrderedDict, deque

import gymnasium as gym
import numpy as np
from numpy import array, float32

# noinspection PyUnresolvedReferences
import racecar_gym.envs.gym_api
from racecar_gym.bullet.positioning import RandomPositioningWithinBallStrategy
from racecar_gym.speed_profile import SpeedProfile


class RaceEnvV2(gym.Env):
    """RaceEnv with a corner-aware reward and a hard-corner curriculum.

    Differences from racecar_gym.env.RaceEnv:

    Reward
    - The throttle bonus (r_motor) is removed. It rewarded full throttle everywhere and, combined with the
      braking delay, taught the agent to brake randomly or to crash at the hairpin at full speed.
    - Progress is rewarded per meter driven.
    - A braking envelope (see SpeedProfile) gives the maximum safe speed at every point of the track. Driving
      faster than the envelope is penalized in proportion to the excess speed, so the agent is taught to
      brake BEFORE a corner instead of after hitting the wall.
    - A wall crash ends the episode with a penalty that grows with speed.

    Curriculum
    - With probability `hard_start_prob` an episode starts a few meters before a sharp corner (the hairpin
      is always among them), or before a place where this environment crashed recently.
    - Other episodes use the original random start.

    Observation and action spaces are identical to RaceEnv, so a trained model is a drop-in replacement.
    """
    camera_name = 'camera_competition'
    motor_name = 'motor_competition'
    steering_name = 'steering_competition'

    def __init__(self,
                 scenario: str = 'austria_competition_collisionStop',
                 render_mode: str = 'rgb_array_birds_eye',
                 reset_when_collision: bool = False,
                 max_steps: int = 2500,
                 # reward parameters
                 k_progress: float = 10.0,  # reward per meter of forward progress
                 k_overspeed: float = 1.5,  # penalty per (m/s) above the braking envelope, per step
                 k_steer: float = 0.05,  # penalty per unit of steering change
                 k_stall: float = 0.3,  # penalty per step while standing still
                 stall_speed: float = 0.3,  # below this speed (m/s) the car counts as standing still
                 stall_limit: int = 150,  # consecutive standing-still steps (3 s) that end the episode
                 stall_end_penalty: float = 10.0,
                 checkpoint_bonus: float = 0.5,
                 crash_penalty_base: float = 5.0,
                 crash_penalty_speed: float = 2.0,  # extra crash penalty per m/s
                 wrong_way_penalty: float = 5.0,
                 # curriculum parameters
                 hard_start_prob: float = 0.5,
                 crash_replay_prob: float = 0.5,
                 approach_progress=(0.06, 0.16),  # start this far (in progress) before the corner
                 profile: SpeedProfile = None,
                 **kwargs):
        self.scenario = scenario.upper()[0] + scenario.lower()[1:]
        self.env_id = f'SingleAgent{self.scenario}-v0'
        self.env = gym.make(id=self.env_id,
                            render_mode=render_mode,
                            reset_when_collision=reset_when_collision,
                            **kwargs)
        self.render_mode = render_mode
        self.action_space = gym.spaces.box.Box(low=-1., high=1., shape=(2,), dtype=float32)
        observation_spaces = {k: v for k, v in self.env.observation_space.items()}
        assert self.camera_name in observation_spaces, \
            f'One of the sensors must be {self.camera_name}. Check the scenario file.'
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 128, 128), dtype=np.uint8)

        self.max_steps = max_steps
        self.k_progress = k_progress
        self.k_overspeed = k_overspeed
        self.k_steer = k_steer
        self.k_stall = k_stall
        self.stall_speed = stall_speed
        self.stall_limit = stall_limit
        self.stall_end_penalty = stall_end_penalty
        self.checkpoint_bonus = checkpoint_bonus
        self.crash_penalty_base = crash_penalty_base
        self.crash_penalty_speed = crash_penalty_speed
        self.wrong_way_penalty = wrong_way_penalty

        self.hard_start_prob = hard_start_prob
        self.crash_replay_prob = crash_replay_prob
        self.approach_progress = approach_progress
        self.profile = profile if profile is not None else SpeedProfile()
        self.corner_entries = self.profile.corner_entries()
        self.recent_crashes = deque(maxlen=50)

        self.cur_step = 0
        self.stall_steps = 0
        self.prev_steer = 0.0
        self.prev_progress = 0.0
        self.prev_checkpoint = 0

    # ------------------------------------------------------------------ helpers
    def observation_postprocess(self, obs):
        return obs[self.camera_name].astype(np.uint8).transpose(2, 0, 1)

    def _sample_start_progress(self):
        """Return a progress value to start from, or None for the default random start."""
        if random.random() >= self.hard_start_prob:
            return None
        if self.recent_crashes and random.random() < self.crash_replay_prob:
            anchor = random.choice(self.recent_crashes)
        elif self.corner_entries:
            anchor = random.choice(self.corner_entries)
        else:
            return None
        return (anchor - random.uniform(*self.approach_progress)) % 1.0

    def _start_pose(self, center_progress):
        world = self.env.unwrapped._scenario.world
        maps = world._maps
        strategy = RandomPositioningWithinBallStrategy(progress_map=maps['progress'],
                                                       obstacle_map=maps['obstacle'],
                                                       drivable_map=maps['occupancy'].map,
                                                       progress_center=center_progress,
                                                       progress_radius=0.01,
                                                       min_distance_to_obstacle=0.5)
        return strategy.get_pose(agent_index=0)

    @staticmethod
    def _fix_checkpoint(infos):
        if infos["checkpoint"] == 5:
            infos["checkpoint"] = 4

    # ------------------------------------------------------------------ gym api
    def reset(self, *args, **kwargs: dict):
        options = dict(kwargs.get('options') or {})
        options['mode'] = 'random'
        center = self._sample_start_progress()
        if center is not None:
            try:
                options['start_pose'] = self._start_pose(center)
            except ValueError:
                pass  # no valid pose near this progress; fall back to the random start
        kwargs['options'] = options

        obs, info = self.env.reset(*args, **kwargs)
        obs = self.observation_postprocess(obs)

        self.cur_step = 0
        self.stall_steps = 0
        self.prev_steer = 0.0
        self.prev_progress = info["progress"]
        self._fix_checkpoint(info)
        self.prev_checkpoint = info["checkpoint"]
        return obs, info

    def step(self, actions):
        self.cur_step += 1
        motor_action, steering_action = actions

        # Small noise, as in the original environment.
        motor_action = np.clip(motor_action + np.random.normal(scale=0.001), -1., 1.)
        steering_action = np.clip(steering_action + np.random.normal(scale=0.01), -1., 1.)
        dict_actions = OrderedDict([(self.motor_name, array(motor_action, dtype=float32)),
                                    (self.steering_name, array(steering_action, dtype=float32))])

        obs, _, _, _, infos = self.env.step(dict_actions)
        obs = self.observation_postprocess(obs)
        self._fix_checkpoint(infos)

        # Forward progress of this step in meters (handles the wrap-around at the start line).
        delta = infos["progress"] - self.prev_progress
        if delta < -0.5:
            delta += 1.0
        elif delta > 0.5:
            delta -= 1.0
        meters = delta * self.profile.lap_length

        speed = float(np.linalg.norm(np.asarray(infos["velocity"])[:2]))
        allowed = self.profile.allowed_speed(infos["progress"])
        overspeed = max(0.0, speed - allowed)

        reward = self.k_progress * meters
        reward -= self.k_overspeed * overspeed
        reward -= self.k_steer * abs(steering_action - self.prev_steer)
        # Standing still must always be worse than trying: otherwise parking in front of a corner
        # (no crash, tiny penalty) beats driving through it.
        stalled = speed < self.stall_speed
        self.stall_steps = self.stall_steps + 1 if stalled else 0
        if stalled:
            reward -= self.k_stall
        if self.prev_checkpoint != infos["checkpoint"] and meters > 0:
            reward += self.checkpoint_bonus

        crashed = bool(infos["wall_collision"])
        wrong_way = bool(infos["wrong_way"])
        if crashed:
            reward -= self.crash_penalty_base + self.crash_penalty_speed * speed
            self.recent_crashes.append(infos["progress"])
        elif wrong_way:
            reward -= self.wrong_way_penalty

        stall_end = self.stall_steps >= self.stall_limit
        if stall_end:
            reward -= self.stall_end_penalty
        terminates = crashed or wrong_way or stall_end
        truncates = self.cur_step >= self.max_steps

        infos["allowed_speed"] = allowed
        infos["speed"] = speed
        infos["overspeed"] = overspeed

        self.prev_steer = steering_action
        self.prev_progress = infos["progress"]
        self.prev_checkpoint = infos["checkpoint"]

        return obs, reward, terminates, truncates, infos

    def render(self):
        return self.env.render()
