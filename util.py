import os
from lib.util import Initialization, Loss, Reward
import torch

from enum import IntEnum, unique

@unique
class REUSE(IntEnum) :
    NONE=1
    PARTIAL=2
    FULL=3


class DOLS_NR :
    def __init__(self,w , v):
        self.w = w
        self.v = v
        self.learned = False
        self.currentV = 0


def getFileName(algorithm, WEIGHT_INITIALIZATION, WEIGHT_MU, LOSS_TYPE, LEARNING_RATE_ACTOR, LEARNING_RATE_CRITIC,  GAMMA, LAMDA, BATCH_SIZE, REPLAY_SIZE, TAU,
                LEARNING_ITER, ACTION_SPACE, SC_RADIUS, La, DEMAND_UE_DATA_RATE, SENB_ISD, NUM_UE, MODE_FOR_REWARD, REQUIRED_DEMAND_UE_DATA_RATE, WEIGHT, SCALING) :

    sub2_dir = 'NN=[INIT='+Initialization(WEIGHT_INITIALIZATION).name+', MU='+str(WEIGHT_MU) + ', LOSS=' + Loss(LOSS_TYPE).name + ']'
    sub_dir = "LR=(" + str(LEARNING_RATE_ACTOR)+','+str(LEARNING_RATE_CRITIC) + "), GAMMA="+str(GAMMA)+", LAMDA="+str(LAMDA) + ", BS="+str(BATCH_SIZE)+\
              ", RS="+str(REPLAY_SIZE)+", TAU="+str(TAU)+", LI="+str(LEARNING_ITER)+ ", AS="+str(ACTION_SPACE)


    if MODE_FOR_REWARD == Reward.DEFAULT.value :
        rewardType = ", La=" + str(La)
    elif MODE_FOR_REWARD == Reward.SINGLEOBJ.value :
        rewardType = ", REQ_RATE=" + str(REQUIRED_DEMAND_UE_DATA_RATE)
    elif MODE_FOR_REWARD == Reward.MULTIOBJ.value:
        rewardType = ", w="+ str(WEIGHT) + ", SCALING="+str(SCALING)

    file_name = "runs/"+algorithm+"/"+sub_dir+"/"+sub2_dir+"/SC_Radius=" + str(SC_RADIUS)+"M" + rewardType + "/UE_Rate=" + str(DEMAND_UE_DATA_RATE/10**6)+"MB" + ', UE=' + str(NUM_UE)
    if os.path.isdir(file_name):
        i = 0
        new_file_name = file_name + "_" + str(i)
        while os.path.isdir(new_file_name):
            i += 1
            new_file_name = file_name + "_" + str(i)

        return new_file_name
    else :
        return file_name


def adv_normalize(adv):
    std = adv.std()

    assert std != 0. and not torch.isnan(std), 'Need nonzero std'
    n_advs = (adv - adv.mean()) / (adv.std() + 1e-8)
    return n_advs




