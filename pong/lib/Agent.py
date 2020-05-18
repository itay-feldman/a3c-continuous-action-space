import torch
import random
import numpy as np
import torch.nn.functional as F


class ProbabilityActionSelector:
    """
    Converts probabilities of actions into action by sampling them
    """
    def __call__(self, probs):
        assert isinstance(probs, np.ndarray)
        actions = []
        for prob in probs:
            actions.append(np.random.choice(len(prob), p=prob))
        return np.array(actions)


def default_preprocessor(states):
    """
        The default preprocessing function, is used if nothign else
        is specified at decalration of Agent
        Transform states to float32 np array then to torch Tensors
    """
    if len(states) == 1:
        np_states = np.expand_dims(states[0], 0)
    else:
        np_states = np.array([np.array(s, copy=False) for s in states], copy=False)
    return torch.tensor(np_states)


class Agent:
    def __init__(self, net, device='cpu', preprocessor=default_preprocessor, apply_softmax=False, action_selector=ProbabilityActionSelector()):
        self.net = net
        self.device = device
        self.preprocessor = preprocessor
        # self.action_limit = action_limit
        self.apply_softmax = apply_softmax
        self.action_selector = action_selector

    @torch.no_grad()
    def __call__(self, states):
        """
            This function is called when the Agent is called
            Preprocesses the states, passes them to the net
            and calculates the actions
        """
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        
        probs_v = self.net(states)
        if self.apply_softmax:
            probs_v = F.softmax(probs_v, dim=1)
        probs = probs_v.data.cpu().numpy()
        actions = self.action_selector(probs)
        return np.array(actions)

    def initial_state(self):
        return None

