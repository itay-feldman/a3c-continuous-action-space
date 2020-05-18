import numpy as np
import random
from collections import namedtuple
import gym
from collections import deque


# Simple named tuple that stores Experience
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done'])


class ExperienceSource:
    """
        Class that creates an iterable experience source, returns
        an Experience object tuple each iteration
    """
    def __init__(self, envs, agent, steps_count=1):
        """
        Create simple experience source
        :param env: environment or list of environments to be used
        :param agent: callable to convert batch of states into actions to take
        :param steps_count: count of steps to track for every experience chain
        """
        super().__init__()
        assert isinstance(steps_count, int)
        assert steps_count >= 1

        if isinstance(envs, (list, tuple)):
            self.envs = envs
        else:
            self.envs = [envs]
        self.agent = agent
        self.steps_count = steps_count
        self.total_rewards = []
        self.total_steps = []

    def __iter__(self):
        """
            This function is called each time the object is
            iterated over, it uses the agent and  the
            environment to generate Experience
        """
        # Setup
        states, batches, current_rewards, current_steps, env_lengths = [], [], [], [], []
        total_steps = [0] * len(self.envs)
        for env in self.envs:
            env_lengths.append(1)
            states.append(env.reset())
            current_rewards.append(0.0)
            current_steps.append(0)
            batches.append(deque(maxlen=self.steps_count))
        
        while True:
            # main loop
            actions = [None] * len(states)
            states_in, states_idx = [], []
            # get all states that will be passed to the agent (and their indices):
            for idx, state in enumerate(states):
                if state is None:
                    # if environment's firs observation is None, sample a random action
                    actions[idx] = self.envs[0].action_space.sample()
                else:
                    states_in.append(state)
                    states_idx.append(idx)
            if states_in:
                out_actions = self.agent(states_in)
                for idx, action in enumerate(out_actions):
                    global_idx = states_idx[idx]
                    actions[global_idx] = action
            grouped_actions = _group_list(actions, env_lengths)

            global_offset = 0
            for env_idx, (env, action_v) in enumerate(zip(self.envs, grouped_actions)):
                next_state, r, done, _ = env.step(action_v[0])
                next_state_v, r_v, done_v = [next_state], [r], [done]
            
                for offset, (action, next_state, r, done) in enumerate(zip(action_v, next_state_v, r_v, done_v)):
                    idx = global_offset + offset
                    state = states[idx]
                    batch = batches[idx]

                    current_rewards[idx] += r
                    current_steps[idx] += 1

                    if state is not None:
                        batch.append(Experience(state=state, action=action, reward=r, done=done))
                    if len(batch) == self.steps_count:
                        yield tuple(batch)
                    states[idx] = next_state
                    if done:
                        if 0 < len(batch) < self.steps_count:
                            yield tuple(batch)
                        while len(batch) > 1:
                            batch.popleft()
                            yield tuple(batch)
                        self.total_rewards.append(current_rewards[idx])
                        self.total_steps.append(current_steps[idx])
                        current_rewards[idx] = 0.0
                        current_steps[idx] = 0

                        states[idx] = env.reset()
                        batch.clear()
            global_offset += len(action_v)

    def pop_total_rewards(self):
        """
            Return all the rewards accumulated so far
        """
        r = list(self.total_rewards)  # python passes a refernce of a list, so we need to explicitly create a new one
        if r:
            self.total_rewards = []
        return r
    
    def pop_rewards_steps(self):
        """
            Return all the rewards accumulated so far
            paired with the number of steps that episode was
        """
        res = list(zip(self.total_rewards, self.total_steps))
        if res:
            self.total_rewards, self.total_steps = [], []
        return res

def _group_list(items, lens):
    """
    Unflat the list of items by lens
    :param items: list of items
    :param lens: list of integers
    :return: list of list of items grouped by lengths
    """
    res = []
    cur_ofs = 0
    for g_len in lens:
        res.append(items[cur_ofs:cur_ofs+g_len])
        cur_ofs += g_len
    return res


ExperienceFirstLast = namedtuple('ExperienceFirstLast', ['state', 'action', 'reward', 'last_state'])


class ExperienceSourceFirstLast(ExperienceSource):
    """
    This is a wrapper around ExperienceSource to prevent storing full trajectory in replay buffer when we need
    only first and last states. For every trajectory piece it calculates discounted reward and emits only first
    and last states and action taken in the first state.
    
    If we have partial trajectory at the end of episode, last_state will be None
    """
    def __init__(self, env, agent, gamma, steps_count=4):
        assert isinstance(gamma, float)
        super(ExperienceSourceFirstLast, self).__init__(env, agent, steps_count=steps_count+1)
        self.gamma = gamma
        self.steps = steps_count
    
    def __iter__(self):
        for exp in super(ExperienceSourceFirstLast, self).__iter__():
            if exp[-1].done and len(exp) <= self.steps:
                last_state = None
                elems = exp
            else:
                last_state = exp[-1].state
                elems = exp[:-1]
            total_reward = 0.0
            for e in reversed(elems):
                total_reward *= self.gamma
                total_reward += e.reward
            yield ExperienceFirstLast(state=exp[0].state, action=exp[0].action, reward=total_reward, last_state=last_state)
            
            


