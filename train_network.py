import torch
import math
import torch.optim as optim
import parameter as p
from util import adv_normalize
import torch.nn.functional as F
from lib.util import Loss

def calc_logprob(mu_v, logstd_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2*torch.exp(logstd_v).clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))
    return p1 + p2



def train_policy(opt_act, net_act, states_v, action, batch_old_logprob_v, batch_adv_v):
    # actor training
    opt_act.zero_grad()

    mu_v = net_act(states_v)
    logprob_pi_v = calc_logprob(mu_v, net_act.logstd, action)

    ratio_v = torch.exp(logprob_pi_v - batch_old_logprob_v)
    # Normalized Advantages
    # if p.ADV_NORMALIIZAION:
    #     batch_adv_v = adv_normalize(batch_adv_v)

    surr_obj_v = batch_adv_v * ratio_v
    clipped_surr_v = batch_adv_v * torch.clamp(ratio_v, 1.0 - p.PPO_EPS, 1.0 + p.PPO_EPS)
    loss_policy_v = -torch.min(surr_obj_v, clipped_surr_v).mean()

    # Calculate entropy bonus
    # entropy_bonus = net_act.entropies(dist).mean()
    # entropy = -p.ENTROPY_COEFF * entropy_bonus
    # print("b loss_policy_v", loss_policy_v)
    # loss_policy_v = loss_policy_v + entropy
    # print("a loss_policy_v", loss_policy_v)

    loss_policy_v.backward()
    if p.CLIP_GRAD_NORM != -1:
        torch.nn.utils.clip_grad_norm(net_act.parameters(), p.CLIP_GRAD_NORM)
    opt_act.step()

    return ratio_v, surr_obj_v, clipped_surr_v, loss_policy_v

def train_policy_for_dol(opt_act, net_act, states_v, action, batch_old_logprob_v, batch_adv_v, weight):
    # actor training
    opt_act.zero_grad()

    mu_v = net_act(states_v)
    logprob_pi_v = calc_logprob(mu_v, net_act.logstd, action)
    ratio_v = torch.exp(logprob_pi_v - batch_old_logprob_v)

    # Normalized Advantages
    # if p.ADV_NORMALIIZAION:
    #     batch_adv_v = adv_normalize(batch_adv_v)

    print("w", weight)
    weight = torch.FloatTensor([weight])
    print("w", weight, weight.shape)

    batch_adv_v = weight.mul(batch_adv_v)
    print("a batch_adv_v", batch_adv_v, batch_adv_v.shape)

    surr_obj_v = batch_adv_v * ratio_v
    print("surr_obj_v", surr_obj_v, surr_obj_v.shape)
    clipped_surr_v = batch_adv_v * torch.clamp(ratio_v, 1.0 - p.PPO_EPS, 1.0 + p.PPO_EPS)
    loss_policy_v = -torch.min(surr_obj_v, clipped_surr_v).mean()

    # Calculate entropy bonus
    # entropy_bonus = net_act.entropies(dist).mean()
    # entropy = -p.ENTROPY_COEFF * entropy_bonus
    # print("b loss_policy_v", loss_policy_v)
    # loss_policy_v = loss_policy_v + entropy
    # print("a loss_policy_v", loss_policy_v)

    loss_policy_v.backward()
    if p.CLIP_GRAD_NORM != -1:
        torch.nn.utils.clip_grad_norm(net_act.parameters(), p.CLIP_GRAD_NORM)
    opt_act.step()

    return loss_policy_v

def train_value(opt_crt, net_crt, states_v, batch_ref_v) :
    opt_crt.zero_grad()
    value_v = net_crt(states_v)
    if p.LOSS_TYPE == Loss.MSE.value:
        loss_value_v = F.mse_loss(value_v.squeeze(-1), batch_ref_v)
    elif p.LOSS_TYPE == Loss.SMOOTHL1.value:
        loss_value_v = F.smooth_l1_loss(value_v.squeeze(-1), batch_ref_v)
    loss_value_v.backward()
    opt_crt.step()

    return loss_value_v