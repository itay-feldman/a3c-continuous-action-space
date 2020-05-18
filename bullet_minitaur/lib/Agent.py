import torch
import random
import numpy as np
import torch.nn.functional as F


def default_preprocessor(states):
    """
        The default preprocessing function, is used if nothign else
        is specified at decalration of Agent
        Transform states to float32 np array then to torch Tensors
    """
    np_states = np.array(states, dtype=np.float32)
    return torch.tensor(np_states)


class Agent:
    def __init__(self, net, device='cpu', preprocessor=default_preprocessor, action_limit=(-1, 1)):
        self.net = net
        self.device = device
        self.preprocessor = preprocessor
        self.action_limit = action_limit  # clip  the actions

    @torch.no_grad()
    def __call__(self, states):
        """
            This function is called when the Agent is called
            Preprocesses the states, passes them to the net
            and calculates the actions
        """
        if self.preprocessor:
            states_v = self.preprocessor(states)
            if torch.is_tensor(states_v):
                states_v = states_v.to(self.device)

        mu_v, var_v, _ = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        sigma = torch.sqrt(var_v).data.cpu().numpy()
        actions = np.random.normal(mu, sigma)
        actions = np.clip(actions, *self.action_limit)

        return actions
