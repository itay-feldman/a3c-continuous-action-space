import os

import numpy as np
import torch


def unpack_batch(batch, net, last_val_gamma, device='cpu'):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []

    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state, copy=False))
        actions.append(exp.action)
        rewards.append(exp.reward)
        if exp.last_state is not None:  # we have to check this way because of numpy ambiguity
            not_done_idx.append(idx)
            np_last_state = np.array(exp.last_state, copy=False)
            last_states.append(np_last_state)

    states_v = torch.FloatTensor(states).to(device)
    actions_t = torch.LongTensor(actions).to(device)

    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = torch.FloatTensor(last_states).to(device)
        last_vals_v = net(last_states_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += last_val_gamma * last_vals_np
    
    vals_ref_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_t, vals_ref_v

