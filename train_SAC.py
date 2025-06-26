import os
import wandb
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback
from webots_remote_env import WebotsRemoteEnv
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


def make_env(port,i):
    def _init():

        env = WebotsRemoteEnv(port)
        if(i==0):
            env = Monitor(env)  # For logging with wandb
        
        return env
    return _init

class CustomWandbLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        if not infos:
            return True

        info = infos[0]
        done = dones[0]
        log_data = {}


        step_keys = [
            "episode",
            "reward_step",
            "avg_speed",
            "final_distance_to_target",
            "mean_proximity_reward",
            "proximity_to_obstacle",
            "collision_flag"
        ]

        for key in step_keys:
            if key in info:
                log_data[key] = info[key]

        if done and "episode_result" in info:

            episode_keys = [
                "distance_covered",
                "trajectory_variance",
                "efficiency_score",
                "episode_result",
                "successes_every_50"
            ]

            for key in episode_keys:
                if key in info:
                    log_data[key] = info[key]

        if(log_data):
            wandb.log(log_data)

        return True



if __name__ == '__main__':

    wandb.init(
        project="RL_tesla_project_UDR",          
        name="SAC-2M-UDR-Webots-run-02", 
        sync_tensorboard=True,             
        monitor_gym=True,                  
        save_code=True
    )
    wandb.define_metric("episode_type/*", step_metric="episode")


    ports = list(range(10000, 10010)) #10000 - 10009
    envs = SubprocVecEnv([make_env(p,i) for i,p in enumerate(ports)])

    CHECKPOINT_DIR = "checkpoint_dir/SAC_2M_UDR"
    MODEL_DIR = "model_dir/SAC_2M_UDR"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,                     
        save_path=CHECKPOINT_DIR,
        name_prefix="SAC_2M_UDR_"
    )

    # Wandb callback
    wandb_callback = WandbCallback(
        gradient_save_freq=0,
        model_save_path=MODEL_DIR,
        verbose=2,
    )


    try:
        print("Training with: SAC")
        model = SAC(MlpPolicy, envs, verbose=1, device='cuda', tensorboard_log="./tb_logs/")
        print(f'Using device: {model.device}')

        
        model.learn(
            total_timesteps=2_000_000,
            progress_bar=True,
            callback=[checkpoint_callback, wandb_callback,CustomWandbLogger()]
        )

        # Salva modello finale
        model.save(os.path.join(MODEL_DIR, "SAC_2M_UDR.mdl"))

    finally:
        envs.close()
        wandb.finish()
