# RL_project_final
## Project Abstract

A custom reinforcement learning environment is developed in Webots, where an autonomous vehicle
navigates a deterministic track with randomly positioned obstacles. The setup evaluates control policies under varying conditions while preserving a fixed
route structure. Training is conducted using two
state-of-the-art continuous control algorithms: Soft
Actor-Critic (SAC) and Proximal Policy Optimization (PPO). To improve generalization, Uniform Domain Randomization (UDR) is applied to obstacle positions and properties, introducing controlled variability across episodes. This strategy enhances robustness
by exposing the agent to diverse configurations during
training. Comparative results show gains in training stability, sample efficiency, and generalization to
unseen scenarios, highlighting UDR’s advantage over
static environments. The findings support the role
of domain randomization in sim-to-real transfer for
autonomous driving.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Folder Structure](#folder-structure)
- [How It Works](#how-it-works)
- [Training Instructions](#training-instructions)
- [Evaluation](#evaluation)
- [Logging and Monitoring](#logging-and-monitoring)
- [Troubleshooting](#troubleshooting)
- [Author](#author)

---

## Features

- Webots-based Tesla simulation environment
- Custom Gymnasium-compatible Python environment
- Socket-based communication between Webots and Python
- Reinforcement Learning with Stable-Baselines3 (e.g., PPO, SAC)
- Parallel training via `SubprocVecEnv`
- Reward shaping for safe driving and obstacle avoidance
- Model checkpointing and logging (W&B + TensorBoard)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Polixide/RL_project_final.git
cd RL_project_final
```

### 2. Create and Activate a Virtual Environment (Optional but Recommended)
```
python3 -m venv venv
source venv/bin/activate      
# On Windows: venv\Scripts\activate
```
### 3. Install All Python Dependencies
```
pip install -r requirements.txt
```
### 4. Install Webots 2025
Go to the official Cyberbotics website: https://cyberbotics.com/#download ,
or use the direct link:
```
wget https://github.com/cyberbotics/webots/releases/download/R2025a/webots-R2025a-x86-64.tar.bz2
```
Extract and install:
```
tar -xjf webots-R2025a-x86-64.tar.bz2
sudo mv webots /opt/webots
```
You can now run Webots using:
```
webots
```


## Folder Structure
```bash
RL_project_final/
├── controllers/              # Webots controllers (e.g., tesla_controller.py)
├── worlds/                   # Webots simulation world (.wbt files)
├── model_dir/                # Final saved models (e.g., best_model.zip)
├── checkpoint_dir/           # Intermediate checkpoints (only contents ignored)
│   └── .gitkeep
├── tb_logs/                  # TensorBoard logs (only contents ignored)
│   └── .gitkeep
├── wandb/                    # Weights & Biases logs (only contents ignored)
│   └── .gitkeep
├── venv/                   # your virtual environment with all the dependencies
│  
├── webots_remote_env.py      # Webots socket communication handler
├── train_PPO.py              # PPO training script
├── train_SAC.py              # SAC training script
├── test_PPO.py               # PPO evaluation script
├── test_SAC.py               # SAC evaluation script
├── run_PPO.sh                # Shell script to launch PPO training
├── run_SAC.sh                # Shell script to launch SAC training
├── requirements.txt          # All Python dependencies
└── README.md                 # Project documentation

```
## How It Works
- Webots runs a 3D world with a Tesla robot.

- The robot uses tesla_controller.py which acts as a socket server.

- Python scripts (e.g., train.py) act as socket clients and communicate with the controller.

- The RL agent sends reset, step, and exit commands.

- Observations are gathered from sensors (e.g., distances, camera).

- Rewards guide learning to avoid obstacles and drive safely.

- Logging is done via W&B and TensorBoard.

## Training Instructions



Start Training
```
# For PPO training
./run_PPO.sh

# For SAC training
./run_SAC.sh
```
Training uses multiple environments in parallel via SubprocVecEnv. You can adjust the number of instances and hyperparameters inside train.py.

## Evaluation
Step 1 — Open Webots
Launch the simulation:


```
webots worlds/your_world.wbt
```
**WARNING: Make sure your robot (Tesla) uses tesla_controller.py as the controller.**

Step 2 - Launch the test.py file
```
python test_ALGO.py 
```
**You can see the car movements in webots while the test is running.**

## Logging and Monitoring
Before first use:
```
wandb login
```
Training logs and plots are available on https://wandb.ai under your account.

## Troubleshooting

- GitHub Password Auth Failed:
Use a Personal Access Token (PAT) instead of your GitHub password.
→ Create Token

- Folders Not Being Pushed to Git:
Add a .gitkeep file inside empty folders that are otherwise ignored by .gitignore.

- Socket Connection Error:
Ensure Webots is running and listening on the correct port.
You can set PORT manually in your environment or use the default (e.g., 10000).

## Authors

- Daniele Catalano (@Polixide) - Politecnico di Torino , Data Science & Engineering
- Samuele Caruso (@Knightmare2002) - Politecnico di Torino , Data Science & Engineering

- Francesco DalCero (@Dalceeee) - Politecnico di Torino , Data Science & Engineering

- Ramadan Mehmetaj (@Danki02) - Politecnico di Torino , Data Science & Engineering
