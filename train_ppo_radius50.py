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
from lib.experienceSource import ExperienceSource
from util import getFileName, adv_normalize
from train_network import train_value, train_policy
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
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


def calc_adv_ref(trajectory, net_crt, states_v, GAMMA, GAE_LAMBDA, device="cpu"):
    """
    By trajectory calculate advantage and 1-step ref value
    :param trajectory: trajectory list
    :param net_crt: critic network
    :param states_v: states tensor
    :return: tuple with advantage numpy array and reference values
    """
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
            delta = exp.reward + GAMMA * next_val - val
            last_gae = delta + GAMMA * GAE_LAMBDA * last_gae
        result_adv.append(last_gae)
        result_ref.append(last_gae + val)
    adv_v = torch.FloatTensor(list(reversed(result_adv))).to(device)
    ref_v = torch.FloatTensor(list(reversed(result_ref))).to(device)
    return adv_v, ref_v


def start(env, La) :
    env.setLa(La)
    env.setRewardType(Reward.DEFAULT.value)
    net_act = ppoModel.ModelActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    net_crt = ppoModel.ModelCritic(env.observation_space.shape[0]).to(device)

    file_name = getFileName("PPO(Radius)", p.WEIGHT_INITIALIZATION, p.WEIGHT_MU, p.LOSS_TYPE, p.LEARNING_RATE_ACTOR, p.LEARNING_RATE_CRITIC, p.GAMMA, p.GAE_LAMBDA, p.BATCH_SIZE, p.REPLAY_SIZE,
                            p.TAU, p.LEARNING_ITER, p.ACTION_SPACE, p.SC_RADIUS, La, p.DEMAND_UE_DATA_RATE, p.SENB_ISD, p.NUM_UE,
                            Reward.DEFAULT.value, p.REQUIRED_DEMAND_UE_DATA_RATE, "")
    writer = SummaryWriter(file_name)

    agent = ppoModel.AgentA2C(net_act, device=device)
    exp_source = ExperienceSource(env, agent, steps_count=1)
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
            writer.add_scalar("WeightPowerConsumption", env.getWeightObj1(La[0]), step_idx)
            writer.add_scalar("WeightDateRate", env.getWeightObj2(La[1]), step_idx)
            writer.add_scalar("R", exp[0].reward, step_idx)
            if rewards_steps:
                rewards, steps = zip(*rewards_steps)
                writer.add_scalar("episode_steps", np.mean(steps), step_idx)
                tracker.reward(np.mean(rewards), step_idx)

            trajectory.append(exp)
            if len(trajectory) < p.TRAJECTORY_SIZE:
                continue


            print("{}-action min, max: {}, {}".format(step_idx, actionInfo[0], actionInfo[1]))
            print("{}-action min, max count: {}, {}".format(step_idx, actionInfo[2], actionInfo[3]))
            print("{}-action std: {}".format(step_idx, actionInfo[4]))
            print("{}-action : {}".format(step_idx, actionInfo[5]))
            print('{}-UePath : {}'.format(step_idx, env.sim.getUePath()))
            print('{}-UeLink : {}'.format(step_idx, env.sim.getUeInLink()))
            print('{}-blocking : {}'.format(step_idx, env.getBlockingPro()))
            print("{}-actor std : {}".format(step_idx, np.mean(net_act.logstd.data.cpu().numpy())))
            # shuffle
            #print("t", trajectory)
            #np.random.shuffle(trajectory)
            #print("t", trajectory)

            traj_states = [t[0].state for t in trajectory]
            traj_actions = [t[0].action for t in trajectory]
            traj_states_v = torch.FloatTensor(traj_states).to(device)
            traj_actions_v = torch.FloatTensor(traj_actions).to(device)
            #old_logprob_v = torch.FloatTensor(traj_log_probs).to(device)
            traj_adv_v, traj_ref_v = calc_adv_ref(trajectory, net_crt, traj_states_v, p.GAMMA, p.GAE_LAMBDA, device=device)
            mu_v = net_act(traj_states_v)
            old_logprob_v = calc_logprob(mu_v, net_act.logstd, traj_actions_v)

            count = 0
            for i in traj_adv_v :
                if i < 0 :
                    count +=1

            print("the number of -adv_count :{}/{}".format(count, len(traj_adv_v)))
            # drop last entry from the trajectory, an our adv and ref value calculated without it
            trajectory = trajectory[:-1]
            old_logprob_v = old_logprob_v[:-1].detach()

            sum_loss_value = 0.0
            sum_loss_policy = 0.0
            count_steps = 0

            total_ratio = []
            total_surr_obj_v = []
            total_clipped_surr_v = []
            t1 = []
            t2 = []
            t3 = []
            t4 = []

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

                    if batch_adv_v.mean().item() < 0:
                        t1.append(np.mean(ratio_v.data.cpu().numpy()))
                        t3.append(batch_adv_v.mean().item())
                    else:
                        t2.append(np.mean(ratio_v.data.cpu().numpy()))
                        t4.append(batch_adv_v.mean().item())

                    total_ratio.append(np.mean(ratio_v.data.cpu().numpy()))
                    total_surr_obj_v.append(np.mean(surr_obj_v.data.cpu().numpy()))
                    total_clipped_surr_v.append(np.mean(clipped_surr_v.data.cpu().numpy()))

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
            writer.add_scalar("std", np.mean(net_act.logstd.data.cpu().numpy()), step_idx)
            #writer.add_scalar("total_ratio", np.mean(total_ratio), step_idx)
            #writer.add_scalar("total_surr_obj_v", np.mean(total_surr_obj_v), step_idx)
            #writer.add_scalar("total_clipped_surr_v", np.mean(total_clipped_surr_v), step_idx)

            print("advantage", traj_adv_v.mean().item())
            print("total_ratio", np.mean(total_ratio))
            print("total_surr_obj_v", np.mean(total_surr_obj_v))
            print("total_clipped_surr_v", np.mean(total_clipped_surr_v))

            if len(t1) != 0 :
                print("t1", np.mean(t1), len(t1))
                print("t3", np.mean(t3), len(t3))
            if len(t2) !=0 :
                print("t2", np.mean(t2), len(t2))
                print("t4", np.mean(t4), len(t4))

            if len(tracker.total_rewards) >= 209 :
                return





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help='Enable CUDA')
    parser.add_argument("-n", "--name", required=False, default="PPO", help="Name of the run")
    parser.add_argument("-s", "--save", default=True, help="Save env")
    parser.add_argument("-l", "--load", default=False, help="Load env")

    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    save_path = os.path.join("saves", "ppo-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    if args.load == True :
        with open('env/save/'+ENV_ID+'('+str(p.NUM_UE)+'),('+str(50)+').pickle', 'rb') as f:
            sim = pickle.load(f)
        env = gym.make(ENV_ID)
        observation_space, action_space = env.load_sim(sim)
        env.observation_space = observation_space
        env.action_space = action_space
    else :
        env = gym.make(ENV_ID)
        if args.save == True :
            copy_sim = {}
            for k, v in env.sim.__dict__.items():
                if k != 'log':
                    copy_sim[k] = v

            with open('env/save/'+ENV_ID+'('+str(p.NUM_UE)+'),('+str(50)+').pickle', 'wb') as fi:
                pickle.dump(copy_sim, fi, pickle.HIGHEST_PROTOCOL)

    test_env = env
    print("env.observation_space.shape[0] : ", env.observation_space.shape[0])
    print("env.action_space.shape[0] : ", env.action_space.shape[0])

    #La = [[0.4,0.6], [0.3, 0.7], [0.2,0.8], [0.8, 0.2], [0.7,0.3], [0.6,0.4]]
    start(env, [0.4,0.6])