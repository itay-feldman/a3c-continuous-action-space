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

# Constants
GAMMA = 0.99
LEARNING_RATE = 1e-6
ENTROPY_BETA = 1e-4
BATCH_SIZE = 16
OPTIM_EPS = 1e-3

STEPS_COUNT = 4  # TODO: try to have several envs in each process
MAX_STEPS = 900
DO_CLIP_GRAD = True
CLIP_GRAD = 0.1

PROCESSES_COUNT = 4  # TODO: what's better? 4 procs and 1 env each, or 2 procs and 4 envs each
NUM_ENVS = 1
REWARD_BOUNDRY = 10  # TODO change

NAME = 'carla'
LOAD_MODEL = None  # 'models/best_rgb.pt'

TotalReward = namedtuple('TotalReward', field_names='reward')


def make_env():
    # TODO might be good to change the car type, the mustang seems to have some physics issues on collision sometimes
    return CarlaEnv(max_steps=MAX_STEPS, img_type='rgb', img_x=480, img_y=360, img_fov=200, traffic_light=False, calc_lane_invasion=False, calc_velocity_limit=False, calc_distace=True, calc_lane_type=True, calc_velocity=False, calc_rotation=True, vehicle="vehicle.chevrolet.impala", fixed_delta_seconds=round(1/30, 3))


def calc_logprob(mu_v, var_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2*var_v.clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * var_v))
    return p1 + p2


def data_func(net, device, train_queue):
    envs = [make_env() for _ in range(NUM_ENVS)]
    agent = Agent(net, device=device)
    # TODO compare training with rgb to semantic and other variations
    exp_source = ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=STEPS_COUNT)
    
    print(f'{mp.current_process().name} Started')

    for exp in exp_source:
        new_rewards = exp_source.pop_total_rewards()
        # print('New rewards', new_rewards)
        if new_rewards:
            train_queue.put(TotalReward(reward=np.mean(new_rewards)))
            # print('Pop goes the weasel!')
        train_queue.put(exp)


def main():
    # some setyp
    mp.set_start_method('spawn')

    # writer
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if LOAD_MODEL:
        name = f'runs/{NAME}_a3c_continued_{timestr}'
    else:
        name = f'runs/{NAME}_a3c_{timestr}'
    writer = SummaryWriter()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using:', device)
    
    env = make_env()
    obs_shape = env.observation_space.shape
    print('Observation shape:', obs_shape)
    act_space = 2
    print('Action space:', act_space)

    if LOAD_MODEL:
        net = torch.load(LOAD_MODEL)
        print('Model loaded from:', LOAD_MODEL)
    else:
        net = ModelA3C(obs_shape, act_space)
    net = net.to(device)

    env.close()  # our env creates new actors that we don't need, we erase them here
    net.share_memory()  # enabled by default for CUDA, but needs to be enabled explicitly for CPU

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=OPTIM_EPS)  # TODO test different epsilon

    train_queue = mp.Queue(maxsize=PROCESSES_COUNT)
    data_proc_list = []

    for _ in range(PROCESSES_COUNT):
        data_proc = mp.Process(target=data_func, args=(net, device, train_queue))
        data_proc.start()
        data_proc_list.append(data_proc)
    
    batch = []
    time_step = 0

    # add current hyperparameters to TensorBoard
    hparams = {'gamma': GAMMA, 'lr': LEARNING_RATE, 'entropy_beta': ENTROPY_BETA,
        'baIMtch_size': BATCH_SIZE, 'steps_count': STEPS_COUNT, 'optim_epsilon': OPTIM_EPS}
    if DO_CLIP_GRAD:
        hparams['clip_grad_threshhold'] = CLIP_GRAD
    writer.add_hparams(hparams, {})
    
    try:
        start_time = time.time()
        print(f'Training Started - {datetime.datetime.now()}')
        with tracking.RewardTracker(writer, REWARD_BOUNDRY) as tracker:
            with tracking.TBMeanTracker(writer, batch_size=10) as tb_tracker:
                while True:
                    train_entry = train_queue.get()
                    if isinstance(train_entry, TotalReward):
                        if tracker.reward(train_entry.reward, time_step):
                            break
                        continue

                    time_step += 1
                    batch.append(train_entry)
                    if len(batch) < BATCH_SIZE:
                        continue
                        
                    states_v, actions_v, vals_ref_v = unpack_batch(batch, net, last_val_gamma=GAMMA**STEPS_COUNT, device=device)
                    batch.clear()
                    # print('batch', states_v.shape, actions_v.shape, vals_ref_v.shape)

                    optimizer.zero_grad()
                    mu_v, var_v, value_v = net(states_v)
                    # print('net', mu_v.shape, var_v.shape, value_v.shape)
                    loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                    adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
                    log_prob_v = adv_v * calc_logprob(mu_v, var_v, actions_v)  # .unsqueeze(-1))
                    loss_policy_v = -log_prob_v.mean()
                    entropy_loss_v = ENTROPY_BETA * (-(torch.log(2*math.pi*var_v) + 1)/2).mean()

                    loss_v = loss_policy_v + entropy_loss_v + loss_value_v
                    loss_v.backward()
                    optimizer.step()

                    tb_tracker.track("advantage", adv_v, time_step)
                    tb_tracker.track("values", value_v, time_step)
                    tb_tracker.track("batch_rewards", vals_ref_v, time_step)
                    tb_tracker.track("loss_entropy", entropy_loss_v, time_step)
                    tb_tracker.track("loss_policy", loss_policy_v, time_step)
                    tb_tracker.track("loss_value", loss_value_v, time_step)
                    tb_tracker.track("loss_total", loss_v, time_step)
                    
        # save model when training ends
        print(f'\nConvergence reached! Solved in {round(time.time() - start_time, 3)} seconds')
        save_path = f'models/a3c_{NAME}_{timestr}.pt'
        torch.save(net.cpu(), save_path)
        print('Saved model to:', save_path)
    
    except KeyboardInterrupt:
        print('Stopped by the user')
        save_path = f'models/a3c_{NAME}_stopped_{timestr}.pt'
        torch.save(net.cpu(), save_path)
        print('Saved model to:', save_path)

    except Exception as e:
        print('Training Crushed:')
        traceback.print_exc()
        save_path = f'models/a3c_{NAME}_error_{timestr}.pt'
        torch.save(net.cpu(), save_path)
        print('Saved model to:', save_path)

    finally:
        # writer.flush()
        for p in data_proc_list:
            p.terminate()
            p.join()
        
        torch.cuda.empty_cache()

        client = carla.Client('localhost', 2000)
        client.reload_world()  # to erase everything from last run
                
            
if __name__ == "__main__":
    main()
