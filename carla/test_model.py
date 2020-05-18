#!/usr/bin/env python3
import numpy as np
from collections import namedtuple
import time
import math
import datetime
import gym
import gym.spaces
import traceback

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torchvision

from lib.experience import ExperienceSourceFirstLast
from lib.CarlaEnv import CarlaEnv
from lib.Agent import Agent
from model import ModelA3C
from lib.utils import unpack_batch
from lib import tracking

import carla

ENV_NAME = "MinitaurBulletEnv-v0"
LOAD_MODEL = './models/best_rgb.pt'
FRAME_RATE = 60  # 60 frames per second, frame=1/60


def make_env():
    return CarlaEnv(max_steps=MAX_STEPS, img_type='rgb', img_x=480, img_y=360, img_fov=200, traffic_light=False, calc_lane_invasion=False, calc_velocity_limit=False, calc_distace=True, calc_lane_type=True, calc_velocity=False, calc_rotation=True, vehicle="vehicle.chevrolet.impala", fixed_delta_seconds=round(1/30, 3))


def main():
    """
        After training is done we test and watch our model perform
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = torch.load(LOAD_MODEL)
    net = net.to(device)
    net.eval()  # set model into evaluation mode
    agent = Agent(net, device=device)
    print('Model loaded from:', LOAD_MODEL)

    env = make_env()
    state = [env.reset()]

    rewards_history, current_rewards = [], []
    episode_counter = 0

    try:
        while True:
            env.render()
            time.sleep(1/FRAME_RATE)
            if state is not None:
                action = agent(state)
            else:
                action = env.action_space.sample()  # if obs is none, sample random action
            next_state, reward, done, _ = env.step(action[0])
            state = [next_state]
            current_rewards.append(reward)
            if done:
                episode_counter += 1
                print('Episode', episode_counter, 'Done.')  # Mean Reward:', np.mean(current_rewards))
                rewards_history.append(current_rewards)
                current_rewards.clear()
                print('Starting Next Episode...')
                state = [env.reset()]

    except KeyboardInterrupt:
        print('Stopped By The User')
        print('Exiting...')

        client = carla.Client('localhost', 2000)
        client.reload_world()


if __name__ == "__main__":
    main()

