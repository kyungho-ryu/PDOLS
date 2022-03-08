#!/usr/bin/env python3
import os
import math
import ptan
import time
import gym
import env
import pickle
import argparse
import parameter as p
from tensorboardX import SummaryWriter
from lib import ppoModel
from lib.util import Loss
from lib.experienceSource import ExperienceSourceOFDOL
from util import getFileName, adv_normalize, REUSE
from train_network import train_value, train_policy
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
from lib.util import Reward
ENV_ID = "hetNetEnv-v0"

def test_net(net, env, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)[0]
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            #action = np.clip(action, -1, 1)
            action = np.clip(action + 1, 0, 2)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


def calc_logprob(mu_v, logstd_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2*torch.exp(logstd_v).clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))
    return p1 + p2


def calc_adv_ref(trajectory, net_crt, states_v, weight, device="cpu"):
    """
    By trajectory calculate advantage and 1-step ref value
    :param trajectory: trajectory list
    :param net_crt: critic network
    :param states_v: states tensor
    :return: tuple with advantage numpy array and reference values
    """
    weight = torch.FloatTensor([weight])

    values_v = net_crt(states_v)
    values = values_v.squeeze().data.cpu().numpy()
    # generalized advantage estimator: smoothed version of the advantage
    last_gae = 0.0
    result_adv = []
    result_ref = []
    for val, next_val, (exp,) in zip(reversed(values[:-1]), reversed(values[1:]),
                                     reversed(trajectory[:-1])):
        if exp.done:
            delta = exp.reward - val
            last_gae = delta
        else:
            delta = exp.reward + p.GAMMA * next_val - val
            last_gae = delta + p.GAMMA * p.GAE_LAMBDA * last_gae

        temp = torch.FloatTensor(last_gae).unsqueeze(-1)
        result_adv.append(weight.matmul(temp))
        result_ref.append(last_gae + val)

    adv_v = torch.FloatTensor(list(reversed(result_adv))).to(device)
    ref_v = torch.FloatTensor(list(reversed(result_ref))).to(device)

    return adv_v, ref_v

# nepisodes = number of episodes
def start(env, weight, model, n_episodes=500) :
    current_episode = 0
    # Set weight for env
    #env.set_weightOBJ(weight)
    env.setRewardType(Reward.MULTIOBJ.value)

    net_act = ppoModel.ModelActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    net_crt = ppoModel.ModelCriticForDol(env.observation_space.shape[0], len(weight)).to(device)

    file_name = getFileName("DOL", p.WEIGHT_INITIALIZATION, p.WEIGHT_MU, p.LOSS_TYPE, p.LEARNING_RATE_ACTOR, p.LEARNING_RATE_CRITIC, p.GAMMA, p.GAE_LAMBDA, p.BATCH_SIZE, p.REPLAY_SIZE,
                            p.TAU, p.LEARNING_ITER, p.ACTION_SPACE, p.SC_RADIUS, p.La, p.DEMAND_UE_DATA_RATE, p.SENB_ISD, p.NUM_UE,
                            Reward.MULTIOBJ.value, p.REQUIRED_DEMAND_UE_DATA_RATE, list(weight), p.SCALING)
    writer = SummaryWriter(file_name)

    agent = ppoModel.AgentA2C(net_act, device=device)
    exp_source = ExperienceSourceOFDOL(env, agent, steps_count=1)
    #exp_source = ptan.experience.ExperienceSource(env, agent, steps_count=1)

    opt_act = optim.Adam(net_act.parameters(), lr=p.LEARNING_RATE_ACTOR)
    opt_crt = optim.Adam(net_crt.parameters(), lr=p.LEARNING_RATE_CRITIC)

    trajectory = []
    best_reward = None
    update_idx = 1

    with ptan.common.utils.RewardTracker(writer) as tracker:
        for step_idx, exp in enumerate(exp_source):
            rewards_steps = exp_source.pop_rewards_steps()
            actionInfo = env.testGetAction()

            AlPower, BhPower, PowerSaving = env.getPowerConsumption()
            writer.add_scalar("AlPower", AlPower, step_idx)
            writer.add_scalar("BhPower", BhPower, step_idx)
            writer.add_scalar("PowerSaving", PowerSaving, step_idx)
            writer.add_scalar("dataRate", env.getDataRate(), step_idx)
            writer.add_scalar("Blocking", env.getBlockingPro(), step_idx)
            writer.add_scalar("ActiveLink", env.getActiveLink(), step_idx)
            writer.add_scalar("ActiveSenbs", env.getActiveSenbs(), step_idx)
            writer.add_scalar("WeightPowerConsumption", env.getWeightObj1(p.SCALING), step_idx)
            writer.add_scalar("WeightDateRate", env.getWeightObj2(p.SCALING), step_idx)

            if rewards_steps:
                current_episode +=1
                rewards, steps = zip(*rewards_steps)
                #writer.add_scalar("episode_steps", np.mean(steps), step_idx)
                #tracker.reward(np.mean(rewards), step_idx)
                r1 = np.mean(rewards[0][::2])
                r2 = np.mean(rewards[0][1::2])
                writer.add_scalar("reward_1", r1, step_idx)
                writer.add_scalar("reward_2", r2, step_idx)
                writer.add_scalar("reward_total", r1+r2, step_idx)

                #print("{}: done {} episodes [{}, {}/{}]".format(step_idx, current_episode, r1, r2, r1+r2))

            trajectory.append(exp)
            if len(trajectory) < p.TRAJECTORY_SIZE:
                continue


            #print("{}-action min, max: {}, {}".format(step_idx, actionInfo[0], actionInfo[1]))
            #print("{}-action min, max count: {}, {}".format(step_idx, actionInfo[2], actionInfo[3]))
            #print("{}-action std: {}".format(step_idx, actionInfo[4]))
            #print("{}-action : {}".format(step_idx, actionInfo[5]))
            # print('{}-UePath : {}'.format(step_idx, env.sim.getUePath()))
            # print('{}-UeLink : {}'.format(step_idx, env.sim.getUeInLink()))
            print('{}-blocking : {}'.format(step_idx, env.getBlockingPro()))

            traj_states = [t[0].state for t in trajectory]
            traj_actions = [t[0].action for t in trajectory]
            traj_states_v = torch.FloatTensor(traj_states).to(device)
            traj_actions_v = torch.FloatTensor(traj_actions).to(device)
            #old_logprob_v = torch.FloatTensor(traj_log_probs).to(device)
            traj_adv_v, traj_ref_v = calc_adv_ref(trajectory, net_crt, traj_states_v, weight, device=device)
            mu_v = net_act(traj_states_v)
            old_logprob_v = calc_logprob(mu_v, net_act.logstd, traj_actions_v)

            # drop last entry from the trajectory, an our adv and ref value calculated without it
            trajectory = trajectory[:-1]
            old_logprob_v = old_logprob_v[:-1].detach()

            sum_loss_value = 0.0
            sum_loss_policy = 0.0
            count_steps = 0

            for epoch in range(p.LEARNING_ITER):
                for batch_ofs in range(0, len(trajectory), p.PPO_BATCH_SIZE):
                    states_v = traj_states_v[batch_ofs:batch_ofs + p.PPO_BATCH_SIZE]
                    actions_v = traj_actions_v[batch_ofs:batch_ofs + p.PPO_BATCH_SIZE]
                    batch_adv_v = traj_adv_v[batch_ofs:batch_ofs + p.PPO_BATCH_SIZE].unsqueeze(-1)
                    batch_ref_v = traj_ref_v[batch_ofs:batch_ofs + p.PPO_BATCH_SIZE]
                    batch_old_logprob_v = old_logprob_v[batch_ofs:batch_ofs + p.PPO_BATCH_SIZE]

                    # critic training
                    loss_value_v = train_value(opt_crt, net_crt, states_v, batch_ref_v)
                    # actor training
                    ratio_v, surr_obj_v, clipped_surr_v, loss_policy_v = train_policy(opt_act, net_act, states_v, actions_v, batch_old_logprob_v, batch_adv_v)

                    sum_loss_value += loss_value_v.item()
                    sum_loss_policy += loss_policy_v.item()
                    count_steps += 1

                    writer.add_scalar("loss_policy", loss_policy_v.item(), update_idx)
                    writer.add_scalar("loss_value", loss_value_v.item(), update_idx)

                    update_idx+=1

            trajectory.clear()
            writer.add_scalar("advantage", traj_adv_v.mean().item(), step_idx)
            writer.add_scalar("values", traj_ref_v.mean().item(), step_idx)
            writer.add_scalar("average loss_policy", sum_loss_policy / count_steps, step_idx)
            writer.add_scalar("average loss_value", sum_loss_value / count_steps, step_idx)

            if current_episode >= n_episodes :
                V = net_crt(torch.FloatTensor(env.reset()).to(device))
                return V.data, (net_act, net_crt)




# returns model and V_PI
# V_PI is the value vector obtained by solving the MOMDP at w and
# current_model is the model at the end of learning
def solveSODP(env, weight, Models, reuse_flag) :
    initialModel = None
    #if reuse_flag :
    #    initialModel = mostSimilarModel(weight, Models)

    V, model = start(env, weight, Models)

    return V, model


def isIncluded(S, V_PI):
    print("isIncluded S : {}, V : {}".format(S, V_PI))
    for s in S :
        if s[0] == V_PI[0] and s[1] == V_PI[1] :
            return True

    return False


def intersection(p1, p2, p3, p4):
    denominator = ((p1[0] - p2[0]) * (p3[1] - p4[1]))\
                  - ((p1[1] - p2[1]) * (p3[0] - p4[0]))

    numeratorX = (((p1[0] * p2[1]) - (p1[1] * p2[0])) * (p3[0] - p4[0]))\
                 - ((p1[0] - p2[0]) * ((p3[0] * p4[1]) - (p3[1] * p4[0])))

    numeratorY = (((p1[0] * p2[1]) - (p1[1] * p2[0])) * (p3[1] - p4[1]))\
                 - ((p1[1] - p2[1]) * ((p3[0] * p4[1]) - (p3[1] * p4[0])))

    return numeratorX / denominator, numeratorY / denominator


def hasImprovement(w, V_PI, S, VToPoints):
    print("hasImprovement : {}".format(V_PI))
    if len(S) == 0 or w[1] == float('inf') :
        return True

    w = w[0]
    currentHeight = None

    for s in S :
        Vi = VToPoints[s]
        print("vi : {} from S".format(Vi))
        if Vi[0][0] == w :
            currentHeight = Vi[0][1]
            break
    print("current Height : {}".format(currentHeight))

    x, y = intersection(p1=(w, 0), p2=(w, currentHeight),
                        p3=(0, V_PI[0]), p4=(1, V_PI[1]))

    print("x, y : {}, {}".format(x, y))
    if y > currentHeight : return True
    else : return False

# V_PI, S, W, VToPoints
# Removes value vectors made obselete by the new V_PI.
def removeObseleteValueVectors(V_PI, S, W, VToPoints):
    obseleteValueVectors = []
    for i in range(len(list(S))-1, -1,-1) :
        range_vi = VToPoints[S[i]]
        s = range_vi[0][0]
        e = range_vi[1][0]
        w1_s = s
        w0_s = 1 - w1_s
        w1_e = e
        w0_e = 1 - w1_e
        if w0_s * V_PI[0] + w1_s * V_PI[1] > w0_s * S[i][0] + w1_s * S[i][1]  and \
            w0_e * V_PI[0] + w1_e * V_PI[1] > w0_e * S[i][0] + w1_e * S[i][1] :
            obseleteValueVectors.append(S[i])

            print("removeObseleteValueVectors : {} from S".format(S[i]))
            del VToPoints[S[i]]
            del S[i]



    return obseleteValueVectors

# S :deque([tensor([-0.7267,  0.3968]),
# tensor([-1.1125,  1.7501]),
# tensor([-0.7915,  0.6467]),
# tensor([-1.4462,  1.8808]), (x)
# tensor([-1.1524,  1.7855])])

def newCornerWeights(V_PI, W_DEL, S, VToPoints):
    p1 = VToPoints[V_PI][0]
    p2 = VToPoints[V_PI][1]
    y_axis_intersect = VToPoints[V_PI][2]
    CornerWeights = []
    for Vi in S :
        p3 = VToPoints[Vi][0]
        p4 = VToPoints[Vi][1]
        current_y_axis_intersect = VToPoints[Vi][2]
        cornerW, Y = intersection(p1, p2, p3, p4)

        # checks if the new corner weight is in the range of the two lines
        # Redundant sets of the points back
        if not (cornerW > p2[0] or cornerW < p1[0] or cornerW > p4[0] or cornerW < p3[0]) :
            if y_axis_intersect > current_y_axis_intersect :
                p3[0] = cornerW
                p3[1] = Y
                p2[0] = cornerW
                p2[1] = Y
                VToPoints[V_PI][1] = p2
                VToPoints[Vi][0] = p3
            else :
                p4[0] = cornerW
                p4[1] = Y
                p1[0] = cornerW
                p1[1] = Y
                VToPoints[V_PI][0] = p1
                VToPoints[Vi][1] = p4

            CornerWeights.append([cornerW, None])

    return CornerWeights


# Q, VToPoints[V_PI][1][1], VToPoints[V_PI][2][1]
def removeObseleteWeights(Q, s, e):
    # checks if a weight has w1 between the range of s and e
    # and in that case it deletes it

    print("removeObseleteWeights / s:{}, e:{}".format(s, e))
    print("original Q : {}".format(Q))

    for i in range(len(list(Q))-1, -1, -1) :
        if (Q[i][0] < e and Q[i][0] > s) and Q[i][1] < float('inf') :
            #Q.remove(q)
            del Q[i]

    print("after Q : {}".format(Q))


# estimates the improvement of a corner weight
def estimateImprovement(cornerWeight, S, VToPoints):
    startingVector = None
    endingVector = None

    for s in S :
        Vi = VToPoints[s]
        if Vi[0][0] == cornerWeight : startingVector = Vi
        elif Vi[1][0] == cornerWeight : endingVector = Vi

    firstPoint = endingVector[0]
    cornerPoint = startingVector[0]
    lastPoint = startingVector[1]

    print("first, corner, last : {}, {}, {}".format(firstPoint, cornerPoint, lastPoint))

    # changed height calculation to be with the point corner-weigt and 100
    _, height = intersection(firstPoint, lastPoint, (cornerWeight, 100), cornerPoint)

    # removed the division to handle the case when the value vector is negative
    return (height - cornerPoint[1]) # / cornerPoint[1]

# inserts w in the priority queue according to its improvement value
# w must be in the format (w1, improvement)
def enqueueWeight(Q, cornerWeight):
    Q.appendleft(cornerWeight)
    print("enqueueWeight original : {}".format(Q))
    for i in range(1, len(Q)) :
        if Q[i][1] > Q[i-1][1] :
            temp = Q[i-1]
            Q[i-1] = Q[i]
            Q[i] = temp
        else :
            break
    print("enqueueWeight after : {}".format(Q))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help='Enable CUDA')
    parser.add_argument("-n", "--name", required=False, default="PPO", help="Name of the run")
    parser.add_argument("-s", "--save", default=True, help="Save env")
    parser.add_argument("-l", "--load", default=False, help="Load env")
    parser.add_argument("-lv", "--load_value", default=False, help="Load env")

    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    save_path = os.path.join("saves", "ppo-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    if args.load == True:
        with open('env/save/' + ENV_ID + '(' + str(p.NUM_UE) + ').pickle', 'rb') as f:
            sim = pickle.load(f)
        env = gym.make(ENV_ID)
        observation_space, action_space = env.load_sim(sim)
        env.observation_space = observation_space
        env.action_space = action_space
    else:
        env = gym.make(ENV_ID)
        if args.save == True:
            copy_sim = {}
            for k, v in env.sim.__dict__.items():
                if k != 'log':
                    copy_sim[k] = v

            with open('env/save/' + ENV_ID + '(' + str(p.NUM_UE) + ').pickle', 'wb') as fi:
                pickle.dump(copy_sim, fi, pickle.HIGHEST_PROTOCOL)

    test_env = env
    print("env.observation_space.shape[0] : ", env.observation_space.shape[0])
    print("env.action_space.shape[0] : ", env.action_space.shape[0])


    # multi-objective DRL
    #   m : environment
    #   r : improvement threshold
    #   template : DQN architecture

    # S : empty partial CSS
    # W : empty list of explored corner weights
    # Q : priority queue initialised with the extrema weights simplex with infinite priority
    # DQN_Models : empty table of DQNs, indexed by the weight, w, for which it was learnt
    S = deque()
    W = deque()
    Q = deque()
    Models = {}
    rejectedWeights = deque()
    VToPoints = {}

    # used to control the minimum-improvement value for a weight to be used by DOL/DOL-R.
    minimum_improvement = 0

    # used to control the peek weights used by DOL/DOL-R.
    maxWeight = 1

    if args.load_value :
        pass
    else :
        # setting initial weights (w1, improvement)
        Q.appendleft([maxWeight, float('inf')])
        Q.appendleft([1-maxWeight, float('inf')])

    # while + max iteration
    while len(Q) != 0:
        w = Q.popleft()
        print("current w : {}".format(w))

        # current model(actor, critic)
        V_PI, current_model = solveSODP(env, [1 - w[0], w[0]], Models, p.REUSE_TYPE)

        print("V", V_PI)
        dist = 1 + V_PI[1] - V_PI[0]
        print("dist", dist)
        V_PI = V_PI / dist
        print("new V", V_PI)
        #print("current_model", current_model)

        p1, p2 = [0, V_PI[0]], [1, V_PI[1]]
        W.append([w, V_PI])
        print("p1, p2 : ", p1, p2)
        print("W : ", W)

        if not isIncluded(S, V_PI) and hasImprovement(w, V_PI, S, VToPoints) :
            Models[str(w)] = current_model # should I store the model for any w even if it is not the one accepted

            #Stores the range for which V_PI is optimal in the current CCS,
            #and it's intercept with the y-axis
            VToPoints[V_PI] = [p1, p2, V_PI[0]]
            print("VToPoints : {}".format(VToPoints))
            W_DEL = [w]

            # Removes value vectors made obselete
            obseleteValueVectors = removeObseleteValueVectors(V_PI, S, W, VToPoints)
            print("obseleteValueVectors : ", obseleteValueVectors)

            W_V_PI = newCornerWeights(V_PI, W_DEL, S, VToPoints)
            print("after VToPoints : ", VToPoints)
            print("newCornerWeights : ", W_V_PI)
            S.append(V_PI)

            removeObseleteWeights(Q, VToPoints[V_PI][0][0], VToPoints[V_PI][1][0])
            for i in range(len(W_V_PI)) :
                cornerWeight = W_V_PI[i]
                cornerWeight[1] = estimateImprovement(cornerWeight[0], S, VToPoints)
                print("cornerW :", cornerWeight)
                if cornerWeight[1] > minimum_improvement :
                    enqueueWeight(Q, cornerWeight)
                else :
                    rejectedWeights.append(cornerWeight)

            for point in VToPoints.values():
                print("cornerW : {}/{}".format(point[0], point[1]))
            print("S :{}".format(S))

            print("==================================================================")

    for point in VToPoints.values():
        print("cornerW : {}/{}".format(point[0], point[1]))
    print("S :{}".format(S))
    print("W :{}".format(W))
    print("Q :{}".format(Q))
    print("VToPoints :{}".format(VToPoints))
    print("rejectedWeights : {}".format(rejectedWeights))


