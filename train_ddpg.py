#!/usr/bin/env python3
import os
import ptan
import time
import gym
import env
import pickle
import argparse, copy
from tensorboardX import SummaryWriter
import numpy as np
from util import getFileName
from lib import ddpgModel, common
from lib.util import setDDPGActionSpace
import parameter as p
import torch
import torch.optim as optim
import torch.nn.functional as F


ENV_ID = "hetNetEnv-v0"
# GAMMA = 0.99
# BATCH_SIZE = 64
# LEARNING_RATE = 1e-4
# REPLAY_SIZE = 50000
# REPLAY_INITIAL = 10
# TAU =1 - 1e-3
# TEST_ITERS = 1000
# LEARNING_ITER = 10
# ACTION_NOISE = True
# ACTION_SPACE = 2

def test_net(net, env, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = setDDPGActionSpace(action)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help='Enable CUDA')
    parser.add_argument("-n", "--name", default="DDPG", help="Name of the run")
    parser.add_argument("-s", "--save", default=False, help="Save env")
    parser.add_argument("-l", "--load", default=True, help="Load env")

    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    save_path = os.path.join("saves", "ddpg-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    if args.load == True :
        with open('env/save/'+ENV_ID+'.pickle', 'rb') as f:
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

            with open('env/save/'+ENV_ID+'.pickle', 'wb') as fi:
                pickle.dump(copy_sim, fi, pickle.HIGHEST_PROTOCOL)

    test_env = env
    print("env.observation_space.shape[0] : ", env.observation_space.shape[0])
    print("env.action_space.shape[0] : ", env.action_space.shape[0])

    act_net = ddpgModel.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    crt_net = ddpgModel.DDPGCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    tgt_act_net = ptan.agent.TargetNet(act_net)
    tgt_crt_net = ptan.agent.TargetNet(crt_net)

    file_name = getFileName("DDPG", p.WEIGHT_INITIALIZATION, p.WEIGHT_MU, p.LOSS_TYPE, p.LEARNING_RATE_ACTOR, p.LEARNING_RATE_CRITIC, p.GAMMA, p.BATCH_SIZE, p.REPLAY_SIZE,
                            p.TAU, p.LEARNING_ITER, p.ACTION_SPACE, p.SC_RADIUS, p.La, p.DEMAND_UE_DATA_RATE, p.SENB_ISD)
    writer = SummaryWriter(file_name)

    agent = ddpgModel.AgentDDPG(act_net, device=device, ou_enabled=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=p.GAMMA, steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=p.REPLAY_SIZE)
    act_opt = optim.Adam(act_net.parameters(), lr=p.LEARNING_RATE_ACTOR)
    crt_opt = optim.Adam(crt_net.parameters(), lr=p.LEARNING_RATE_CRITIC)

    frame_idx = 0
    learning_idx = 0

    best_reward = None

    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=1) as tb_tracker:
            while True:
                frame_idx += 1
                buffer.populate(1)
                rewards_steps = exp_source.pop_rewards_steps()

                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", steps[0], frame_idx)
                    tracker.reward(rewards[0], frame_idx)

                if frame_idx % 10000 == 0 :
                    actionInfo = env.testGetAction()
                    print("{}-action min, max: {}, {}".format(frame_idx, actionInfo[0], actionInfo[1]))
                    print("{}-action min, max count: {}, {}".format(frame_idx, actionInfo[2], actionInfo[3]))
                    print("{}-action std: {}".format(frame_idx, actionInfo[4]))
                    print("{}-action : {}".format(frame_idx, actionInfo[5]))
                    print('{}-UePath : {}'.format(frame_idx, env.sim.getUePath()))
                    print('{}-UeLink : {}'.format(frame_idx, env.sim.getUeInLink()))
                    print('{}-blocking : {}'.format(frame_idx, env.getBlockingPro()))

                if len(buffer) < p.REPLAY_INITIAL:
                    continue

                AlPower, BhPower, PowerSaving = env.getPowerConsumption()
                tb_tracker.track("AlPower", AlPower, frame_idx)
                tb_tracker.track("BhPower", BhPower, frame_idx)
                tb_tracker.track("PowerSaving", PowerSaving, frame_idx)
                tb_tracker.track("dataRate", env.getDataRate(), frame_idx)
                tb_tracker.track("Blocking", env.getBlockingPro(), frame_idx)
                tb_tracker.track("ActiveLink", env.getActiveLink(), frame_idx)
                tb_tracker.track("ActiveSenbs", env.getActiveSenbs(), frame_idx)
                tb_tracker.track("WeightPowerConsumption", env.getWeightObj1(), frame_idx)
                tb_tracker.track("WeightDateRate", env.getWeightObj2(), frame_idx)


                for _ in range(p.LEARNING_ITER) :
                    batch = buffer.sample(p.BATCH_SIZE)
                    states_v, actions_v, rewards_v, dones_mask, last_states_v = common.unpack_batch_ddqn(batch, device)

                    # train critic
                    crt_opt.zero_grad()
                    q_v = crt_net(states_v, actions_v)
                    last_act_v = tgt_act_net.target_model(last_states_v)
                    q_last_v = tgt_crt_net.target_model(last_states_v, last_act_v)
                    q_last_v[dones_mask] = 0.0
                    # unsqueeze : Add dim
                    q_ref_v = rewards_v.unsqueeze(dim=-1) + q_last_v * p.GAMMA
                    critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())
                    critic_loss_v.backward()
                    crt_opt.step()
                    tb_tracker.track("q_v", q_v.mean(), learning_idx)
                    tb_tracker.track("loss_critic", critic_loss_v, learning_idx)
                    tb_tracker.track("critic_ref", q_ref_v.mean(), learning_idx)

                    # train actor
                    act_opt.zero_grad()
                    cur_actions_v = act_net(states_v)
                    actor_loss_v = -crt_net(states_v, cur_actions_v)
                    actor_loss_v = actor_loss_v.mean()
                    actor_loss_v.backward()
                    act_opt.step()

                    tb_tracker.track("loss_actor", actor_loss_v, learning_idx)

                    tgt_act_net.alpha_sync(alpha=p.TAU)
                    tgt_crt_net.alpha_sync(alpha=p.TAU)

                    learning_idx +=1

                if frame_idx % p.TEST_ITERS == 0:
                    ts = time.time()
                    rewards, steps = test_net(act_net, test_env, device=device)
                    print("Test done in %.2f sec, reward %.3f, steps %d" % (
                        time.time() - ts, rewards, steps))
                    writer.add_scalar("test_reward", rewards, frame_idx)
                    writer.add_scalar("test_steps", steps, frame_idx)


                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                            name = "best_%+.3f_%d.dat" % (rewards, frame_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(act_net.state_dict(), fname)
                        best_reward = rewards

    pass
