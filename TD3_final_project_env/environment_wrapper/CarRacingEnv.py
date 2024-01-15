import argparse
from collections import deque
import itertools
import random
import time
import cv2

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from racecar_gym.env import RaceEnv
from austria_competition_initial_vision import austria_competition_initial_vision
from circle_cw_competition_collisionStop_initial_vision import circle_cw_competition_collisionStop_initial_vision


class CarRacingEnvironment:
    def __init__(self, N_frame=4):
        env = RaceEnv(
            scenario='circle_cw_competition_collisionStop',  # e.g., 'austria_competition', 'circle_cw_competition_collisionStop'
            render_mode='rgb_array_birds_eye',
            reset_when_collision=False,  # Only work for 'austria_competition' and 'austria_competition_collisionStop'
        )

        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.ep_len = 0
        self.frames = deque(maxlen=N_frame)
        self.env_initial_vision = circle_cw_competition_collisionStop_initial_vision
        self.a_reset_threshold = 10
        self.c_reset_threshold = 609
        self.time_limit = 1250

    '''def check_car_position(self, state):
        # cut the image to get the part where the car is
        part_image = state[60:84, 40:60, :]

        road_color_lower = np.array([90, 90, 90], dtype=np.uint8)
        road_color_upper = np.array([120, 120, 120], dtype=np.uint8)
        grass_color_lower = np.array([90, 180, 90], dtype=np.uint8)
        grass_color_upper = np.array([120, 255, 120], dtype=np.uint8)
        road_mask = cv2.inRange(part_image, road_color_lower, road_color_upper)
        grass_mask = cv2.inRange(part_image, grass_color_lower, grass_color_upper)
        # count the number of pixels in the road and grass
        road_pixel_count = cv2.countNonZero(road_mask)
        grass_pixel_count = cv2.countNonZero(grass_mask)

        # save image for debugging
        # filename = "images/image" + str(self.ep_len) + ".jpg"
        # cv2.imwrite(filename, part_image)

        return road_pixel_count, grass_pixel_count'''

    def step(self, action):
        state, reward, terminates, truncates, info = self.env.step(action)
        state = np.dstack((state[0],state[1],state[2]))
        reward *= 1000
        original_reward = reward
        original_terminates = terminates
        self.ep_len += 1
        '''road_pixel_count, grass_pixel_count = self.check_car_position(state)
        info["road_pixel_count"] = road_pixel_count
        info["grass_pixel_count"] = grass_pixel_count'''

        # convert to grayscale
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)

        if (self.ep_len >= self.time_limit):
            terminates = True

        # my reward shaping strategy, you can try your own
        if info["wall_collision"] or ((self.ep_len >= self.time_limit) and info["progress"] < 1 and info["lap"] == 1):
            terminates = True
            reward = -70

        # save image for debugging
        # filename = "images/image" + str(self.ep_len) + ".jpg"
        # cv2.imwrite(filename, state)

        # frame stacking
        self.frames.append(state)
        state = np.stack(self.frames, axis=0)

        '''if self.test:
            # enable this line to recover the original reward
            reward = original_reward
        # enable this line to recover the original terminates signal, disable this to accerlate evaluation
        # terminates = original_terminates'''

        return state, reward, terminates, truncates, info

    def test_step(self, action):
        state, reward, terminates, truncates, info = self.env.step(action)
        state = np.dstack((state[0],state[1],state[2]))
        reward *= 1000
        self.ep_len += 1
        '''road_pixel_count, grass_pixel_count = self.check_car_position(state)
        info["road_pixel_count"] = road_pixel_count
        info["grass_pixel_count"] = grass_pixel_count'''

        # my reward shaping strategy, you can try your own
        if info["wall_collision"]:
            terminates = True

        # convert to grayscale
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)

        # save image for debugging
        # filename = "images/image" + str(self.ep_len) + ".jpg"
        # cv2.imwrite(filename, state)

        # frame stacking
        self.frames.append(state)
        state = np.stack(self.frames, axis=0)

        return state, reward, terminates, truncates, info

    def reset(self):
        self.frames.clear()
        # check initial vision is clean
        while True:
            state, info = self.env.reset()
            state = np.dstack((state[0], state[1], state[2]))
            # convert to grayscale
            state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
            if np.count_nonzero(state - self.env_initial_vision) < self.c_reset_threshold:
                break
        self.ep_len = 0
        # frame stacking
        for _ in range(self.frames.maxlen):
            self.frames.append(state)
        state = np.stack(self.frames, axis=0)

        return state, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


if __name__ == '__main__':
    env = CarRacingEnvironment(test=True)
    state, info = env.reset()
    done = False
    total_reward = 0
    total_length = 0
    t = 0
    while not done:
        t += 1
        action = env.action_space.sample()
        action[2] = 0.0
        state, reward, terminates, truncates, info = env.step(action)
        print(
            f'{t}: road_pixel_count: {info["road_pixel_count"]}, grass_pixel_count: {info["grass_pixel_count"]}, reward: {reward}')
        total_reward += reward
        total_length += 1
        env.render()
        if terminates or truncates:
            done = True

    print("Total reward: ", total_reward)
    print("Total length: ", total_length)
    env.close()
