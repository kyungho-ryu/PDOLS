import time
import ptan, math
import numpy as np
import torch
import torch.nn as nn
import parameter as p
import torch.nn.functional as F
from torch.distributions import Normal

from lib.util import initNetworkWeight, determinant
from lib.util import setPPOActionSpace

HID_SIZE = 64


class ModelActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ModelActor, self).__init__()

        self.l1 = nn.Linear(obs_size, HID_SIZE)
        self.l2 = nn.Linear(HID_SIZE, HID_SIZE)
        self.mu = nn.Linear(HID_SIZE, act_size)

        initNetworkWeight(p.WEIGHT_INITIALIZATION, self.l1)
        initNetworkWeight(p.WEIGHT_INITIALIZATION, self.l2)
        initNetworkWeight(p.WEIGHT_INITIALIZATION, self.mu)

        self.weightMU = p.WEIGHT_MU
        self.log_stdev = torch.nn.Parameter(torch.zeros(act_size))

    def forward(self, x):
        x = F.tanh(self.l1(x))
        x = F.tanh(self.l2(x))

        #means = self.mu(x)
        mu = torch.tanh(self.mu(x))

        return mu

    def get_loglikelihood(self, p, actions):
        try:
            mean, std = p
            nll =  0.5 * ((actions - mean) / std).pow(2).sum(-1) \
                   + 0.5 * np.log(2.0 * np.pi) * actions.shape[-1] \
                   + self.log_stdev.sum(-1)
            return -nll
        except Exception as e:
            raise ValueError("Numerical error")

    def entropies(self, p):
        '''
        Get entropies over the probability distributions given by p
        p_i = (mean, var), p mean is shape (batch_size, action_space),
        p var is shape (action_space,)
        '''
        _, std = p
        detp = determinant(std)
        d = std.shape[0]
        entropies = torch.log(detp) + .5 * (d * (1. + math.log(2 * math.pi)))
        return entropies

    def sample(self, p):
        '''
        Given prob dist (mean, var), return: actions sampled from p_i, and their
        probabilities. p is tuple (means, var). means shape
        (batch_size, action_space), var (action_space,), here are batch_size many
        prboability distributions you're sampling from

        Returns tuple (actions, probs):
        - actions: shape (batch_size, action_dim)
        - probs: shape (batch_size, action_dim)
        '''
        means, std = p
        # return (means + torch.randn_like(means)*std).detach()
        dist = Normal(means, std)
        actions = dist.sample()
        #return (torch.randn_like(means)*std).detach()
        return actions
    # def forward(self, x, std_change, std_step, softmax_dim = 0):
    #     x = F.relu(self.l1(x))
    #     x = F.relu(self.l2(x))
    #
    #     mu = self.weightMU * torch.tanh(self.mu(x))
    #     #std = F.softplus(self.fc_std(x))
    #     if std_change and self.std - std_step >=0:
    #         self.std -= std_step
    #     return mu, self.std


class ModelCritic(nn.Module):
    def __init__(self, obs_size):
        super(ModelCritic, self).__init__()

        self.value = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1),
        )

    def forward(self, x):
        return self.value(x)


class AgentA2C(ptan.agent.BaseAgent):
    def __init__(self, net, device="cpu"):
        self.net = net
        self.device = device

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states).to(self.device)

        actions = self.net(states_v)
        #next_actions = self.net.sample(dist)
        #print("next_actions", next_actions)

        #next_action_log_probs = self.net.get_loglikelihood(dist, next_actions)
        #next_actions = next_actions.data.cpu().numpy()

        return actions, agent_states
