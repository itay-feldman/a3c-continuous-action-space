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
from lib.Agent import Agent
from model import ModelA3C
from lib.utils import unpack_batch
from lib import tracking
from lib import wrappers

ENV_NAME = "PongNoFrameskip-v4"
LOAD_MODEL = 'models/complete.pt'
FRAME_RATE = 120  # 60 frames per second, frame=1/60


def make_env():
    return wrappers.wrap_dqn(gym.make(ENV_NAME))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = torch.load(LOAD_MODEL)
    net = net.to(device)
    net.eval()  # set model into evaluation mode
    agent = Agent(lambda x: net(x)[0], device=device, apply_softmax=True)
    print('Model loaded from:', LOAD_MODEL)
    print(f'Playing at {FRAME_RATE} FPS')

    env = make_env()
    state = [env.reset()]

    rewards_history, current_rewards = [], []
    game_counter = 0

    try:
        while True:
            env.render()
            time.sleep(1/FRAME_RATE)
            if state is not None:
                action = agent(state)
            else:
                action = env.action_space.sample()  # if obs is none, sample random action
            next_state, reward, done, _ = env.step(action)
            state = [next_state]
            current_rewards.append(reward)
            if done:
                game_counter += 1
                print('Game', game_counter, 'Done.')  # Mean Reward:', np.mean(current_rewards))
                rewards_history.append(current_rewards)
                current_rewards.clear()
                print('Starting Next Game...')
                state = [env.reset()]

    except KeyboardInterrupt:
        print('Stopped By The User')
        print('Exiting...')


if __name__ == "__main__":
    main()

