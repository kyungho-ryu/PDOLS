from lib.util import Initialization, Loss, Reward
from util import REUSE
# Network
GAMMA = 0.6
LEARNING_ITER = 10
LEARNING_RATE_ACTOR = 1e-4 * 1
LEARNING_RATE_CRITIC = 1e-4 * 1
WEIGHT_INITIALIZATION =Initialization.HE.value
WEIGHT_MU = 1.0 # action space [MU * tanh]
LOSS_TYPE = Loss.SMOOTHL1.value

# Env
La = [0.2,0.8]
DEMAND_UE_DATA_RATE = 10**7 * 1.4
REQUIRED_DEMAND_UE_DATA_RATE = 0.6
SENB_ISD = 100
NUM_UE = 100
SC_RADIUS = 80
MODE_FOR_REWARD = Reward.DEFAULT.value # 1 : multi-object, 2 : single-object, 3 : multi-object

## DDPG
BATCH_SIZE = 64
#LEARNING_RATE_ACTOR = 0.0005
#LEARNING_RATE_CRITIC = 0.001
REPLAY_SIZE = 50000
REPLAY_INITIAL = 10000
TAU =1 - 0.005
TEST_ITERS = 1000
ACTION_SPACE = (-10,10)

##PPO
GAE_LAMBDA = 0.7
TRAJECTORY_SIZE = 1025
PPO_EPS = 0.2
#PPO_EPOCHES = 10
PPO_BATCH_SIZE = 32
TEST_ITERS = 100000

##DOL
REUSE_TYPE = REUSE.NONE.value
SCALING = 0.2
ENTROPY_COEFF = 0

# Implementation matters
ADV_NORMALIIZAION = True
ENTROPY_BONUS = True
CLIP_GRAD_NORM = -1
