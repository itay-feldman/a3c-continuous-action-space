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
# from lib.scheduling import Scheduler

# Constants
GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128

STEPS_COUNT = 4
MAX_STEPS = 100
DO_CLIP_GRAD = True
CLIP_GRAD = 0.1

RENDER = True

PROCESSES_COUNT = 4
REWARD_BOUNDRY = 18

ENV_COUNT = 15  # num of env in each process


ENV_NAME = "PongNoFrameskip-v4"
NAME = 'pong'
LOAD_MODEL = './models/latest.pt'

# TODO add argparse for name etc.

TotalReward = namedtuple('TotalReward', field_names='reward')


def make_env():
    return wrappers.wrap_dqn(gym.make(ENV_NAME))


def data_func(net, device, train_queue):
    # TODO might be good to change the car type, the mustagn seems to have some physics issues on collision sometimes
    envs = [make_env() for _ in range(ENV_COUNT)]
    # print('!render', render)
    agent = Agent(lambda x: net(x)[0], device=device, apply_softmax=True)
    # TODO compare training with rgb to semantic and other variations
    exp_source = ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=STEPS_COUNT)
    
    """if scheduler is not None:
        id_num = scheduler.assign_identifier()
        print(f'Process {id_num} Started')"""
    
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
    gym.logger.set_level(40)

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
    act_space = env.action_space.n
    print('Action space:', act_space)

    if LOAD_MODEL:
        net = torch.load(LOAD_MODEL)
        print('Model loaded from:', LOAD_MODEL)
    else:
        net = ModelA3C(obs_shape, act_space)
    net = net.to(device)
    env.close()  # our env creates new actors that we don't need, we erase them here
    net.share_memory()  # enabled by default for CUDA, but needs to be enabled explicitly for CPU

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

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
        'batch_size': BATCH_SIZE, 'steps_count': STEPS_COUNT}
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

                    optimizer.zero_grad()
                    logits_v, value_v = net(states_v)

                    loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                    log_prob_v = F.log_softmax(logits_v, dim=1)
                    adv_v = vals_ref_v - value_v.detach()
                    log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), actions_v]

                    loss_policy_v = -log_prob_actions_v.mean()
                    prob_v = F.softmax(logits_v, dim=1)
                    entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()

                    loss_v = entropy_loss_v + loss_value_v + loss_policy_v
                    loss_v.backward()
                    nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)  # make sure gradiants aren't above certain value (improves stability and convergence time)
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
        save_path = f'models/model_a3c_av_{timestr}.pt'
        torch.save(net.cpu(), save_path)
        print('Saved model to:', save_path)
    
    except KeyboardInterrupt:
        print('Stopped by the user')
        save_path = f'models/model_a3c_av_stopped_{timestr}.pt'
        torch.save(net.cpu(), save_path)
        print('Saved model to:', save_path)

    except Exception as e:
        print('Training Crushed:')
        traceback.print_exc()
        save_path = f'models/model_a3c_av_error_{timestr}.pt'
        torch.save(net.cpu(), save_path)
        print('Saved model to:', save_path)

    finally:
        # writer.flush()
        for p in data_proc_list:
            p.terminate()
            p.join()
        
        torch.cuda.empty_cache()
                
            
if __name__ == "__main__":
    main()
