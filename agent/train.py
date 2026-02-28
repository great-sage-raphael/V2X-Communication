"""
V2X Reinforcement Learning Training Script (train.py)
======================================================
Trains a PPO agent on the V2XEnv Gymnasium environment using Stable-Baselines3.

WHY PPO (not DQN or SAC)?
- DQN requires a flat Discrete action space.
- SAC requires a continuous Box action space.
- PPO natively supports MultiDiscrete, which matches our per-vehicle binary choices.

Directory layout created by this script:
    agent/
      models/
        best_model.zip          ← Best checkpoint (saved by EvalCallback)
        final_model.zip         ← Model after full training run
      logs/
        training_monitor/       ← SB3 Monitor CSV (per-episode reward + length)
        eval_monitor/           ← Evaluation env Monitor CSV
        tb_logs/v2x_ppo_1/      ← TensorBoard event files
        eval_summary.csv        ← Post-training: 50-ep RL policy evaluation
        baseline_summary.csv    ← 50-ep V2I-only rule-based baseline

Usage:
    python agent/train.py                        # full 500k-step run
    python agent/train.py --timesteps 5000       # quick smoke-test
    python agent/train.py --timesteps 500000 --eval-freq 10000
"""

import os
import sys
import argparse
import csv
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
# Allow imports from the project root so `env/v2x_env.py` is discoverable.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "env"))

from v2x_env import V2XEnv

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# ── Directory constants ───────────────────────────────────────────────────────
AGENT_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(AGENT_DIR, "models")
LOGS_DIR     = os.path.join(AGENT_DIR, "logs")
TB_LOGS_DIR  = os.path.join(LOGS_DIR, "tb_logs")

TRAIN_MONITOR_DIR = os.path.join(LOGS_DIR, "training_monitor")
EVAL_MONITOR_DIR  = os.path.join(LOGS_DIR, "eval_monitor")

EVAL_SUMMARY_CSV     = os.path.join(LOGS_DIR, "eval_summary.csv")
BASELINE_SUMMARY_CSV = os.path.join(LOGS_DIR, "baseline_summary.csv")
BEST_MODEL_PATH      = os.path.join(MODELS_DIR, "best_model")
FINAL_MODEL_PATH     = os.path.join(MODELS_DIR, "final_model")


def make_dirs():
    """Create all required output directories."""
    for d in [MODELS_DIR, LOGS_DIR, TB_LOGS_DIR, TRAIN_MONITOR_DIR, EVAL_MONITOR_DIR]:
        os.makedirs(d, exist_ok=True)


def make_env(monitor_dir, tag="train"):
    """
    Factory function that returns a callable creating a monitored V2XEnv.
    Monitor wraps the env and writes per-episode reward + length to a CSV file,
    which the plotting script later reads for the Learning Curve figure.
    """
    def _init():
        env = V2XEnv()
        # Monitor writes: timesteps, episode reward, episode length to monitor.csv
        env = Monitor(env, filename=os.path.join(monitor_dir, tag))
        return env
    return _init


def train_ppo(total_timesteps: int, eval_freq: int):
    """
    Full PPO training loop.

    Architecture:
        - Policy: MlpPolicy with two hidden layers of 256 units.
        - The (20, 4) observation is flattened to an 80-dim input automatically
          by SB3's FlattenObservation feature.
        - EvalCallback: evaluates periodically, saves best model weights.
        - CheckpointCallback: saves a .zip snapshot every 50k steps.
    """
    print("=" * 60)
    print("  V2X Intelligent Communication — PPO Training")
    print(f"  Total timesteps : {total_timesteps:,}")
    print(f"  Eval frequency  : every {eval_freq:,} steps")
    print("=" * 60)

    make_dirs()

    # ── Build Training Environment ─────────────────────────────────────────
    # DummyVecEnv wraps a single env in a vectorised interface required by SB3.
    train_env = DummyVecEnv([make_env(TRAIN_MONITOR_DIR, tag="train")])

    # ── Build Evaluation Environment ───────────────────────────────────────
    # Separate env instance used exclusively by EvalCallback.
    # This ensures evaluation episodes are independent of the training rollout.
    eval_env = DummyVecEnv([make_env(EVAL_MONITOR_DIR, tag="eval")])

    # ── Callbacks ─────────────────────────────────────────────────────────
    # EvalCallback runs `n_eval_episodes` evaluation episodes every `eval_freq`
    # training steps and saves the checkpoint whenever mean reward improves.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = MODELS_DIR,
        log_path             = LOGS_DIR,
        eval_freq            = eval_freq,
        n_eval_episodes      = 5,
        deterministic        = True,
        render               = False,
        verbose              = 1,
    )

    # CheckpointCallback: saves a snapshot every 50k steps as a safety net.
    checkpoint_callback = CheckpointCallback(
        save_freq         = max(50_000, eval_freq * 5),
        save_path         = MODELS_DIR,
        name_prefix       = "v2x_ppo_checkpoint",
        save_replay_buffer= False,
        verbose           = 1,
    )

    # ── PPO Hyperparameters ────────────────────────────────────────────────
    # policy_kwargs: two hidden layers of 256 neurons — sufficient for an
    # 80-dimensional input while avoiding over-parameterisation.
    model = PPO(
        policy          = "MlpPolicy",
        env             = train_env,
        learning_rate   = 3e-4,          # Standard starting LR for PPO
        n_steps         = 2048,          # Steps per rollout buffer
        batch_size      = 64,            # Mini-batch size for policy update
        n_epochs        = 10,            # Gradient update epochs per rollout
        gamma           = 0.99,          # Discount factor
        gae_lambda      = 0.95,          # GAE smoothing factor
        clip_range      = 0.2,           # PPO clipping parameter
        ent_coef        = 0.01,          # Entropy bonus (encourages exploration)
        vf_coef         = 0.5,           # Value function loss weight
        max_grad_norm   = 0.5,
        policy_kwargs   = {"net_arch": [256, 256]},
        tensorboard_log = TB_LOGS_DIR,
        verbose         = 1,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps = total_timesteps,
        callback        = [eval_callback, checkpoint_callback],
        tb_log_name     = "v2x_ppo",
        progress_bar    = True,
    )

    # Save final model regardless of EvalCallback outcome
    model.save(FINAL_MODEL_PATH)
    print(f"\n[✓] Final model saved → {FINAL_MODEL_PATH}.zip")

    # Close envs
    train_env.close()
    eval_env.close()

    return model


def run_evaluation(model, n_episodes=50, label="RL_PPO"):
    """
    Evaluate a trained model for `n_episodes` episodes and write a detailed
    CSV log. Each row captures: episode index, total reward, mean PDR, mean
    latency, V2V ratio, and V2I ratio.

    This CSV is the primary data source for the latency, PDR, and mode
    distribution plots in plot_results.py.
    """
    print(f"\n[→] Running {n_episodes}-episode evaluation: {label}")

    rows = []
    env  = V2XEnv()

    for ep in range(n_episodes):
        obs, _      = env.reset()
        ep_reward   = 0.0
        terminated  = truncated = False

        while not (terminated or truncated):
            # Synchronise RL decisions with SUMO's current vehicle state.
            # The policy produces a (MAX_VEHICLES,) action array; only the
            # first N entries (active vehicles) are used inside env.step().
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

        # Retrieve episode-level aggregated metrics from the env
        summary = env.get_episode_summary()
        rows.append({
            "episode":         ep + 1,
            "total_reward":    round(ep_reward, 4),
            "mean_pdr":        round(summary["mean_pdr"], 4),
            "mean_latency_ms": round(summary["mean_latency_ms"], 4),
            "v2v_ratio":       round(summary["v2v_ratio"], 4),
            "v2i_ratio":       round(summary["v2i_ratio"], 4),
        })

        if (ep + 1) % 10 == 0:
            avg_reward = np.mean([r["total_reward"] for r in rows[-10:]])
            print(f"  Episode {ep+1:3d}/{n_episodes} | Last-10 avg reward: {avg_reward:.3f}")

    env.close()

    # Write CSV
    outfile = EVAL_SUMMARY_CSV if label == "RL_PPO" else BASELINE_SUMMARY_CSV
    _write_csv(rows, outfile)
    print(f"  [✓] Saved → {outfile}")
    return rows


def run_baseline(n_episodes=50):
    """
    Rule-Based Baseline: always choose V2I (mode=0) for every vehicle.

    This policy represents the conventional infrastructure-centric approach
    where vehicles always route through the base station. It provides the
    comparison against which the RL agent's improvements are measured in
    the Latency Comparison and PDR plots.
    """
    print(f"\n[→] Running {n_episodes}-episode V2I-only baseline")

    rows = []
    env  = V2XEnv()

    for ep in range(n_episodes):
        obs, _     = env.reset()
        ep_reward  = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            # Baseline policy: always V2I (action = 0) for all vehicle slots
            action = np.zeros(20, dtype=np.int64)  # All V2I
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

        summary = env.get_episode_summary()
        rows.append({
            "episode":         ep + 1,
            "total_reward":    round(ep_reward, 4),
            "mean_pdr":        round(summary["mean_pdr"], 4),
            "mean_latency_ms": round(summary["mean_latency_ms"], 4),
            "v2v_ratio":       round(summary["v2v_ratio"], 4),
            "v2i_ratio":       round(summary["v2i_ratio"], 4),
        })

        if (ep + 1) % 10 == 0:
            avg_reward = np.mean([r["total_reward"] for r in rows[-10:]])
            print(f"  Episode {ep+1:3d}/{n_episodes} | Last-10 avg reward: {avg_reward:.3f}")

    env.close()
    _write_csv(rows, BASELINE_SUMMARY_CSV)
    print(f"  [✓] Baseline saved → {BASELINE_SUMMARY_CSV}")
    return rows


def _write_csv(rows: list, path: str):
    """Write a list-of-dicts to a CSV file."""
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent on V2XEnv")
    parser.add_argument(
        "--timesteps", type=int, default=500_000,
        help="Total training timesteps (default: 500,000 ≈ 500 episodes)"
    )
    parser.add_argument(
        "--eval-freq", type=int, default=10_000,
        help="EvalCallback frequency in timesteps (default: 10,000)"
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=50,
        help="Number of post-training evaluation episodes (default: 50)"
    )
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="Skip the V2I-only baseline run (saves time on smoke tests)"
    )
    args = parser.parse_args()

    # ── Step 1: Train ──────────────────────────────────────────────────────
    model = train_ppo(args.timesteps, args.eval_freq)

    # ── Step 2: Post-training RL Evaluation ───────────────────────────────
    # Load best model (often better than the final checkpoint)
    best_model_zip = os.path.join(MODELS_DIR, "best_model.zip")
    if os.path.exists(best_model_zip):
        print(f"\n[→] Loading best model from {best_model_zip}")
        model = PPO.load(best_model_zip)

    run_evaluation(model, n_episodes=args.eval_episodes, label="RL_PPO")

    # ── Step 3: Baseline Comparison ───────────────────────────────────────
    if not args.skip_baseline:
        run_baseline(n_episodes=args.eval_episodes)

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Model checkpoints : {MODELS_DIR}/")
    print(f"  Training logs     : {TRAIN_MONITOR_DIR}/train.monitor.csv")
    print(f"  RL eval results   : {EVAL_SUMMARY_CSV}")
    print(f"  Baseline results  : {BASELINE_SUMMARY_CSV}")
    print("  → Run: python src/visualization/plot_results.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
