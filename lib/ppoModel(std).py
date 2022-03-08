import ptan
import numpy as np
import torch
import torch.nn as nn
import parameter as p
import torch.nn.functional as F
from torch.distributions import Normal

from lib.util import initNetworkWeight
from lib.util import setPPOActionSpace

HID_SIZE = 64


class ModelActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ModelActor, self).__init__()

        self.l1 = nn.Linear(obs_size, HID_SIZE)
        self.l2 = nn.Linear(HID_SIZE, HID_SIZE)
        self.mu = nn.Linear(HID_SIZE, act_size)
        self.fc_std = nn.Linear(HID_SIZE, 1)

        initNetworkWeight(p.WEIGHT_INITIALIZATION, self.l1)
        initNetworkWeight(p.WEIGHT_INITIALIZATION, self.l2)
        initNetworkWeight(p.WEIGHT_INITIALIZATION, self.mu)
        initNetworkWeight(p.WEIGHT_INITIALIZATION, self.fc_std)

        # self.mu = nn.Sequential(
        #     nn.Linear(obs_size, HID_SIZE),
        #     nn.Tanh(),
        #     nn.Linear(HID_SIZE, HID_SIZE),
        #     nn.Tanh(),
        #     nn.Linear(HID_SIZE, act_size),
        #     nn.Tanh(),
        # )
        self.weightMU = p.WEIGHT_MU
        self.logstd = nn.Parameter(torch.zeros(act_size))

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))

        mu = self.weightMU * torch.tanh(self.mu(x))
        #std = F.softplus(self.fc_std(x))
        return mu, std

    #def forward(self, x):
    #    return self.mu(x)


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

        #mu_v = self.net(states_v)
        #mu = mu_v.data.cpu().numpy()
        mu, std = self.net.pi(states_v, softmax_dim=1)
        dist = Normal(mu, std)
        actions = dist.sample()
        log_prob = dist.log_prob(actions)
        actions = actions.data.cpu().numpy()
        log_prob = log_prob.data.cpu().numpy()
        #log_prob = dist.log_prob(a)
        # 0
        #logstd = self.net.logstd.data.cpu().numpy()
        #print("aa", np.exp(logstd))

        #actions = mu + np.exp(logstd) * np.random.normal(size=logstd.shape)
        #actions = np.clip(actions, -1, 1)
        #actions = setPPOActionSpace(actions)

        return actions, agent_states, log_prob
