# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 15:40:05 2021

@author: Gachon
"""
from env.hetNet.sim import *
import time, math
from random import uniform
import torch
import numpy as np
import gym
import parameter as p
from lib.util import Reward, new_softmax
from gym import spaces
from gym.utils import seeding
from pkg_resources import parse_version
#	if (argc < 6) {
#		cerr << "Usage: <num SeNB> <num UEs> <ISD> <SC radius> <offset> <x-limit> <y-limit> <hotspot 1-0> <hotspot [0,100]>" << endl;
#		return 1;
#	}
SPACEX =550	#meter
SPACEY =550	#meter

NUM_SENB = 5 #5x5
#NUM_UE = 100
#DEMAND_UE_DATA_RATE = 10**7# 10Mbps
HOTSPOT_UE_RATIO = 0.8
HOTSPOT_UE_DEPLOY = False

## 100 -> Macro - 4, total = 152
## 95 -> Macro -6, total = 216
# 2 25SSSSS
# + 64
#SENB_ISD = 100
OFFSET = 30
#SC_RADIUS = 80
#La =  [1, 1] #multi-object weight

# Macro capcity = 2.7GB
# access link = 223
# bh link = 152
# MaxEnergy = 1033

class HetNetEnergyEnv(gym.Env) :
    def __init__(self, renders=True):
        self.sim = Simul(SPACEX, SPACEY)
        self.testCurrentAction = []
        self.La = [] # for default object
        self.rewardType = []
        self.initSim(self.sim)

        wEdges = {e: uniform(0, p.ACTION_SPACE[1]) for e in self.sim.edges.keys()} #default

        # print("wEdges", wEdges)
        self.updateTopo(self.sim, wEdges)

        observation_dim = len(self.getCurrentState())

        observation_high = np.float32(np.ones([observation_dim]))
        observation_low = np.float32(np.zeros([observation_dim]))

        action_dim = len(wEdges)

        action_high = np.array([np.float32(p.ACTION_SPACE[1])] * action_dim)
        action_low = np.array([np.float32(p.ACTION_SPACE[0])] * action_dim)

        self.action_space = spaces.Box(action_low, action_high, dtype=np.float32)
        self.observation_space = spaces.Box(observation_low, observation_high, dtype=np.float32)

        self._seed()
        self.reset()
        print("access links: ", len(self.sim.anDownlinks))
        print("scell acess links", self.sim.getScellAcessLink())
        print("hotspotUE: ", (self.sim.getHospotUE()))
        print("NotHospotUE: ", self.sim.getNotHospotUE())
        #print("t", [i.id for i in self.sim.hotspotUe])
        print("MaxEnergy : ", self.sim.calMaxEnergy())
        print("LinkCapacity : ", self.sim.getLinkCapa())
        #self.sim.getLinkCapa()
        #self._configure()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._observation = self.getCurrentState()

        return self._observation

    def configSeNBGridTopology(self, netTopo,  n, gap, offset, radius):
        for i in range(n):
            for j in range(n):
                netTopo.SetSeNB(j*gap+ offset, i*gap + offset, radius);

    def initSim(self, sim):
        sim.SetMeNB(10, 10, 1000);
        self.configSeNBGridTopology(sim, NUM_SENB, p.SENB_ISD, OFFSET, p.SC_RADIUS);	# 5x5, gap, offset, radius
        if not HOTSPOT_UE_DEPLOY:
            sim.generateUE(p.NUM_UE, p.SENB_ISD, OFFSET);
        else:
            sim.generateHotSpotUE(p.NUM_UE*HOTSPOT_UE_RATIO)
            sim.generateNoHotSpotUE(p.NUM_UE*(1-HOTSPOT_UE_RATIO))
        sim.ueDataRate = p.DEMAND_UE_DATA_RATE
        #generate links
        sim.findNeighborEnbs(); #between enbs
        sim.findNeighborUEs();  #from enb to ue
        #sim.findReachableEnb(); #from ue to enb
		#print("access links: ", len(sim.anDownlinks))
        # generate topology
        sim.genNxTopo()
        self.getDataRate()

    def load_sim(self, _sim):
        self.sim.load_data(list(_sim.values()))
        self.sim.ueDataRate = p.DEMAND_UE_DATA_RATE

        observation_dim = len(self.getCurrentState())
        observation_high = np.float32(np.ones([observation_dim]))
        observation_low = np.float32(np.zeros([observation_dim]))

        wEdges = {e: uniform(0, p.ACTION_SPACE[1]) for e in self.sim.edges.keys()} #default

        action_dim = len(wEdges)
        action_high = np.array([np.float32(p.ACTION_SPACE[1])] * action_dim)
        action_low = np.array([np.float32(p.ACTION_SPACE[0])] * action_dim)

        self.action_space = spaces.Box(action_low, action_high, dtype=np.float32)
        self.observation_space = spaces.Box(observation_low, observation_high, dtype=np.float32)

        self._seed()
        self.reset()
        print("Change access links: ", len(self.sim.anDownlinks))
        print("Change MaxEnergy : ", self.sim.calMaxEnergy())

        print("Change scell acess links", self.sim.getScellAcessLink())
        print("Change hotspotUE: ", (self.sim.getHospotUE()))
        print("Change NotHospotUE: ", self.sim.getNotHospotUE())

        return self.observation_space, self.action_space

    def updateTopo(self, sim, wEdges): #list of bi-directional edge tuple, wEdge[(1, 4)] = w
        #generate nx topology
        #sim.updateTopo(list(wEdges.keys()))
        sim.findDLInfLinks()
        sim.findBhInfLinks()

        #cal link capa and power
        sim.calLinkCapacity()
        sim.setPossible_mcellU()
        sim.nxTopo.updateTopoWithWeight(wEdges)
        #update routing
        sim.updateUserFlowPath()

    #### functions for RL environment
    def cal_default_Reward(self):
        maxPower = self.sim.calMaxEnergy()

	    #print("max", maxPower)
        #print("t", self.sim.calEnergyConsumption())
        #print("test1",self.sim.calEnergyConsumption()/maxPower)
        #print("test2",sum([(u.flowRate - p.DEMAND_UE_DATA_RATE)/p.DEMAND_UE_DATA_RATE for u in self.sim.ues]))
        #print("test3",([u.flowRate for u in self.sim.ues]))

        #total_reward = -self.La[0] * (self.sim.calEnergyConsumption()/maxPower) + (self.La[1] * (sum([u.flowRate / p.DEMAND_UE_DATA_RATE  for u in self.sim.ues]))/len(self.sim.ues))

        return -self.La[0] * (self.sim.calEnergyConsumption()/maxPower) + (self.La[1] * (sum([u.flowRate / p.DEMAND_UE_DATA_RATE  for u in self.sim.ues]))/len(self.sim.ues))
        #return self.La[0] *(1- (self.sim.calEnergyConsumption()/maxPower)) + (self.La[1] * (sum([u.flowRate / p.DEMAND_UE_DATA_RATE  for u in self.sim.ues]))/len(self.sim.ues))

    def cal_singleOBJ_Reward(self):
        maxPower = self.sim.calMaxEnergy()

        if p.REQUIRED_DEMAND_UE_DATA_RATE > (sum([u.flowRate / p.DEMAND_UE_DATA_RATE for u in self.sim.ues])) / len(self.sim.ues) :
            return 0
        else :
            return (1- (self.sim.calEnergyConsumption() / maxPower))*10

    def set_weightOBJ(self, w):
        self.weight = w

        print("set weight for objects : {}".format(self.weight))

    def cal_MULTIOBJ_Reward(self):
        maxPower = self.sim.calMaxEnergy()

        # EC = (-self.sim.calEnergyConsumption()/maxPower)
        # DR = ((sum([u.flowRate / p.DEMAND_UE_DATA_RATE  for u in self.sim.ues]))/len(self.sim.ues))

        # dist = 1+ DR - EC
        #return [(1-self.sim.calEnergyConsumption()/maxPower)+0.24, ((sum([u.flowRate / p.DEMAND_UE_DATA_RATE  for u in self.sim.ues]))/len(self.sim.ues))]
        return [(-self.sim.calEnergyConsumption()/maxPower) * p.SCALING, ((sum([u.flowRate / p.DEMAND_UE_DATA_RATE  for u in self.sim.ues]))/len(self.sim.ues))* p.SCALING]
        # return [(-self.sim.calEnergyConsumption()/maxPower), ((sum([u.flowRate / p.DEMAND_UE_DATA_RATE  for u in self.sim.ues]))/len(self.sim.ues))]

        # print("or", EC, DR)
        # print("dist", dist)
        # print("re", EC/dist, DR/dist)
        # return [EC / dist, DR/ dist]


    def getWeightObj1(self, w):
        maxPower = self.sim.calMaxEnergy()

        #return w * (1- (self.sim.calEnergyConsumption()/maxPower))
        return - w * ((self.sim.calEnergyConsumption()/maxPower))

    def getWeightObj2(self, w):
        return w*((sum([u.flowRate / p.DEMAND_UE_DATA_RATE  for u in self.sim.ues]))/len(self.sim.ues))

    def getPowerConsumption(self):
        return self.sim.getEnergyConsumption()

    def getBlockingPro(self):
        return self.sim.blocking

    def getDataRate(self):
        return np.sum([u.flowRate for u in self.sim.ues]) / len(self.sim.ues) / 10**6

    def getCurrentState(self):
        return self.sim.getLinksLoad() #link utility from bh to dl access

    def getActiveLink(self) :
        return self.sim.getActiveLink()

    def getActiveSenbs(self):
        return self.sim.getActiveSenbs()

    def getEnvInfo(self):
        return str(p.SC_RADIUS)+"M", str(self.La), str(p.DEMAND_UE_DATA_RATE/10**6)+"MB"

    # delay : 50ms
    def step(self, _wEdges): #action: update weigthed edges, s, a, r, s
        _wEdges = self.setActionSpace(_wEdges, p.ACTION_SPACE)
        self.testCurrentAction = _wEdges
        wEdges = {}
        for i, e in enumerate(self.sim.edges.keys()) :
            wEdges[e] = _wEdges[i]
        #wEdges = {e: wEdges.pop(0) for e in self.sim.edges.keys()}

        self.sim.nxTopo.updateTopoWithWeight(wEdges)
        self.sim.updateUserFlowPath()

        if self.rewardType == Reward.DEFAULT.value:
            r = self.cal_default_Reward()
        elif  self.rewardType == Reward.SINGLEOBJ.value :
            r = self.cal_singleOBJ_Reward()
        elif self.rewardType == Reward.MULTIOBJ.value :
            r = self.cal_MULTIOBJ_Reward()

        #print("reward : ", r)
        s_prime = self.getCurrentState()
        #print("s_prime", s_prime)
		#self.getActiveLink1()
        return s_prime, r, False, {}

    def setActionSpace(self, _wEdges, actionSpace):
        # [0~ 2]
        min_v = min(_wEdges)
        actions = _wEdges - min_v if min_v < 0 else _wEdges

        return actions

    def closeSim(self):
        #save log
        self.sim.log.saveResult()
        self.sim.log.closeLog()

    # if __name__ == "__main__":
    #     print("state :", s)
    #     r = calReward()
    #     print("reward :", r)
    #     closeSim(sim)

    def testGetAction(self):
        min_v = min(list(self.testCurrentAction))
        max_v = max(list(self.testCurrentAction))
        std = np.std(list(self.testCurrentAction))
        return min_v, \
               max_v, \
               list(self.testCurrentAction).count(min_v), \
               list(self.testCurrentAction).count(max_v), \
               std, \
               self.testCurrentAction

    def setLa(self, La):
        self.La = La

        print("set La : {}".format(La))

    def setRewardType(self, type):
        self.rewardType = type

        print("set RewardType : {}".format(self.rewardType))

    def setUEDataRate(self, DEMAND_UE_DATA_RATE):
        self.sim.ueDataRate = DEMAND_UE_DATA_RATE

        print("set ueDataRate : {}".format(self.sim.ueDataRate))