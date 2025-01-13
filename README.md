# **Multi-Objective Optimization of Energy Saving and Throughput in Heterogeneous Networks Using Deep Reinforcement Learning**

Wireless networking using GHz or THz spectra has encouraged mobile service providers to deploy small cells to improve link quality and cell capacity using mmWave backhaul links. However, the high energy consumption of such dense **Heterogeneous Networks (HetNets)** poses a critical challenge, especially in the context of green networking for reducing CO2 emissions.  

This project presents a **dual-objective optimization model** to:  
1. Minimize energy consumption by switching off unused small cells.  
2. Maximize user throughput, formulated as a **Mixed Integer Linear Programming (MILP)** problem.  

Leveraging **Deep Reinforcement Learning (DRL)**, specifically the **Proximal Policy Optimization (PPO)** algorithm with an **Optimistic Linear Support (OLS)** framework, this approach achieves near-optimal solutions with real-time applicability.  

---

## 📜 Research Paper
[**Access the full research paper published in Sensors**](https://www.mdpi.com/1424-8220/21/23/7925)  

---

## 📂 Project Contents

### Directory Structure
- **`/env/hetNet`**  
  Environment setup for HetNet simulation.

- **`/lib`**  
  Implementation of **Actor-Critic models** for PPO-based optimization.

- **`/saves`**  
  Directory for saving:
  - Simulation results
  - Trained models
  - Log files

- **`train_*` Scripts**  
  Scripts for training DRL models:
  - **PPO**
  - **DDPG**
  - **PPO with OLS** (multi-objective optimization)

---

## 🚀 Simulation Overview

### 1. Performance Comparison  
- **Energy Efficiency and Throughput**:  
  The proposed PPO-based algorithm achieves:
  - **28% reduction in energy consumption** compared to CPLEX.
  - Average **user throughput of 13.79 Mbps**, comparable to CPLEX’s 14 Mbps.

  <img src="https://user-images.githubusercontent.com/73271891/157249513-9f2cd549-9d12-48f3-b5a6-f05c07acc190.png" width="50%">

### 2. Reward Weight Exploration  
- Trade-off analysis between energy savings and throughput optimization through reward vector adjustment.  

  <img src="https://user-images.githubusercontent.com/73271891/157249065-d3a71fe5-1c01-40be-99f3-d08c5e072faa.png" width="50%">

---

## 🔧 How to Use

### 1. Environment Setup  
Clone the repository and install dependencies:
```bash
git clone https://github.com/example/HetNetOptimization.git
cd PDOLS
pip install -r requirements.txt
```
### 2. Training
Run training scripts for different algorithms:
```bash
# PPO training
python train_PPO.py

# DDPG training
python train_DDPG.py

# PPO with OLS optimization
python train_PPO_OLS.py
```
