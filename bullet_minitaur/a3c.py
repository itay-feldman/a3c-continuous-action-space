#!/usr/bin/env python3
import numpy as np
from collections import namedtuple
import time
import math
import datetime
import gym
import gym.spaces
import traceback
import pybullet_envs
import pybullet

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torchvision

from .lib.experience import ExperienceSourceFirstLast
from .lib.Agent import Agent, default_preprocessor
from .model import ModelA3C
from .lib.utils import unpack_batch
from .lib import tracking
from .lib import wrappers

# Constants
GAMMA = 0.99
LEARNING_RATE = 5e-5
ENTROPY_BETA = 1e-4
BATCH_SIZE = 32

STEPS_COUNT = 2
MAX_STEPS = 100

RENDER = True

PROCESSES_COUNT = 6
ENV_COUNT = 1  # num of env in each process
TEST_ITERS = 20000  # after n timesteps test the model

ENV_NAME = "MinitaurBulletEnv-v0"
NAME = 'minitaur'
LOAD_MODEL = './models/best.pt'


RewardSteps = namedtuple('RewardSteps', field_names='reward')


def make_env():
    return gym.make(ENV_NAME)


def calc_logprob(mu_v, var_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2*var_v.clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * var_v))
    return p1 + p2


def data_func(net, device, train_queue):
    envs = [make_env() for _ in range(ENV_COUNT)]
    agent = Agent(net=net, device=device)
    exp_source = ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=STEPS_COUNT)
    
    print(f'{mp.current_process().name} Started')

    for exp in exp_source:
        new_rewards = exp_source.pop_rewards_steps()
        if new_rewards:
            train_queue.put(RewardSteps(reward=new_rewards))
        train_queue.put(exp)


def test_net(net, env, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = default_preprocessor([obs]).to(device)
            mu_v = net(obs_v)[0]
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


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
    act_space = env.action_space.shape
    print('Action space:', act_space)

    if LOAD_MODEL:
        net = torch.load(LOAD_MODEL)
        print('Model loaded from:', LOAD_MODEL)
    else:
        net = ModelA3C(obs_shape[0], act_space[0])

    net = net.to(device)
    env.close()  # our env creates new actors that we don't need, we erase them here
    test_env = make_env()  # env to pass to the testing function
    net.share_memory()  # enabled by default for CUDA, but needs to be enabled explicitly for CPU

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    train_queue = mp.Queue(maxsize=PROCESSES_COUNT)
    data_proc_list = []

    for _ in range(PROCESSES_COUNT):
        data_proc = mp.Process(target=data_func, args=(net, device, train_queue))
        data_proc.start()
        data_proc_list.append(data_proc)
    
    batch = []
    best_reward = None
    time_step = 0

    # add current hyperparameters to TensorBoard
    hparams = {'gamma': GAMMA, 'lr': LEARNING_RATE, 'entropy_beta': ENTROPY_BETA,
        'batch_size': BATCH_SIZE, 'steps_count': STEPS_COUNT}
    writer.add_hparams(hparams, {})
    
    try:
        start_time = time.time()
        print(f'Training Started - {datetime.datetime.now()}')
        with tracking.RewardTracker(writer) as tracker:
            with tracking.TBMeanTracker(writer, batch_size=10) as tb_tracker:
                while True:
                    train_entry = train_queue.get()
                    if isinstance(train_entry, RewardSteps):
                        rewards_steps = train_entry.reward
                        rewards, steps = zip(*rewards_steps)
                        tb_tracker.track('episode_steps', steps[0], time_step)
                        tracker.reward(rewards[0], time_step)
                        continue  # wrong type, we don't want total rewards in our batch

                    time_step += 1
                    
                    if time_step % TEST_ITERS == 0:
                        ts = time.time()
                        rewards, steps = test_net(net, test_env, device=device)
                        msg_str = "Test done in %.2f sec, reward %.3f, steps %d" % (time.time() - ts, rewards, steps)
                        if best_reward is not None:
                            msg_str += f' Current Best {round(best_reward, 3)}'
                        print(msg_str)
                        writer.add_scalar("test_reward", rewards, time_step)
                        writer.add_scalar("test_steps", steps, time_step)
                        if best_reward is None or best_reward < rewards:
                            if best_reward is not None:
                                print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                                # name = "best_%+.3f_%d.dat" % (rewards, time_step)
                                save_path = f'models/best_model_a3c_{timestr}.pt'
                                # fname = os.path.join(save_path, name)
                                torch.save(net, save_path)
                            best_reward = rewards

                    
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
        save_path = f'models/model_a3c_{timestr}.pt'
        torch.save(net.cpu(), save_path)
        print('Saved model to:', save_path)
    
    except KeyboardInterrupt:
        print('Stopped by the user')
        save_path = f'models/model_a3c_stopped_{timestr}.pt'
        torch.save(net.cpu(), save_path)
        print('Saved model to:', save_path)

    except Exception as e:
        print('Training Crushed:')
        traceback.print_exc()
        save_path = f'models/model_a3c_error_{timestr}.pt'
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
