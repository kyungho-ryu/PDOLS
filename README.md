# Multi-Objective Optimization of Energy Saving and Throughput in Heterogeneous Networks Using Deep Reinforcement Learning

Wireless networking using GHz or THz spectra has encouraged mobile service providers
to deploy small cells to improve link quality and cell capacity using mmWave backhaul links. As
green networking for less CO2 emission is mandatory to confront global climate change, we need
energy efficient network management for such denser small-cell heterogeneous networks (HetNets)
that already suffer from observable power consumption. We establish a dual-objective optimization
model that minimizes energy consumption by switching off unused small cells while maximizing
user throughput, which is a mixed integer linear problem (MILP). Recently, the deep reinforcement
learning (DRL) algorithm has been applied to many NP-hard problems of the wireless networking
field, such as radio resource allocation, association and power saving, which can induce a nearoptimal solution with fast inference time as an online solution. In this paper, we investigate the
feasibility of the DRL algorithm for a dual-objective problem, energy efficient routing and throughput
maximization, which has not been explored before. We propose a proximal policy (PPO)-based
multi-objective algorithm using the actor-critic model that is realized as an optimistic linear support
framework in which the PPO algorithm searches for feasible solutions iteratively. Experimental results
show that our algorithm can achieve throughput and energy savings comparable to the CPLEX.

Paper : https://www.mdpi.com/1424-8220/21/23/7925


## Contents 
- /env/hetNet 
    - Environment of heterogeneous networks
  
- /lib
    - Actor and Critic model
- /saves 
  - Save simulation results

- train_*
  - DRL model based on PPO or DDPG or PPO with DOL


## Simulation 

- Performance comparison of proposed algorithms
    - Average UE data rate and energy
        <img src="https://github.com/kyungho-ryu/PDOLS/files/8206578/fig_PDOLS.pdf">

    -  Reward weight vector exploration.
        <img src="https://user-images.githubusercontent.com/73271891/157249065-d3a71fe5-1c01-40be-99f3-d08c5e072faa.png">
