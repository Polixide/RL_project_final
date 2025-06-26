import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import wandb

from webots_remote_env import WebotsRemoteEnv

# Initialize WandB for experiment tracking
wandb.init(
    project="RL_project_final",
    name="PPO-2M-test-run_01",
    sync_tensorboard=False,  # Do not sync TensorBoard for test
    monitor_gym=True,
    save_code=False,         # Skip saving the code in WandB
    job_type="eval"          # Label this run as an evaluation job
)

# Path to the trained model
MODEL_DIR = "model_dir/PPO_2M_UDR/PPO_2M_UDR.mdl"

# Number of evaluation episodes
N_EVAL_EPISODES = 30

# --- Load the Webots environment ---
env = WebotsRemoteEnv(10000)

# --- Load the pre-trained model ---
try:
    print(f"Loading model from: {MODEL_DIR}")
    model = PPO.load(MODEL_DIR, env=env, device='cuda')  # Load the model on GPU
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error while loading the model: {e}")
    print("Make sure the model path is correct and the file exists.")
    env.close()
    wandb.finish()
    exit()

# --- Evaluate the model ---
print(f"\nStarting model evaluation for {N_EVAL_EPISODES} episodes...")
try:
    # Run policy evaluation with rendering enabled
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=N_EVAL_EPISODES, render=True)

    print(f"\n--- Evaluation Results ---")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Manually compute average episode length
    episode_lengths = []
    print("\nCollecting episode length data...")
    for i in range(N_EVAL_EPISODES):
        obs, info = env.reset()
        done = False
        episode_len = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)  # Use deterministic actions for evaluation
            obs, reward, done, _, info = env.step(action)
            episode_len += 1
            if done:
                break
        episode_lengths.append(episode_len)
        print(f"Episode {i+1}: Length {episode_len}")

    # Compute episode length statistics
    mean_episode_length = np.mean(episode_lengths)
    std_episode_length = np.std(episode_lengths)
    print(f"Average episode length: {mean_episode_length:.2f} +/- {std_episode_length:.2f}")

    # Log metrics to WandB
    wandb.log({
        "test/mean_reward": mean_reward,
        "test/std_reward": std_reward,
        "test/mean_episode_length": mean_episode_length,
        "test/std_episode_length": std_episode_length
    })

except Exception as e:
    print(f"Error during evaluation: {e}")
    import traceback
    traceback.print_exc()  # Print full traceback for debugging

finally:
    # Clean up
    env.close()
    wandb.finish()
    print("\nTest completed.")
