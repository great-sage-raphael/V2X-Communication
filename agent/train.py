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
        best_model.zip          ← Best checkpoint (saved by CheckpointCallback)
        final_model.zip         ← Model after full training run
      logs/
        training_monitor/       ← SB3 Monitor CSV (per-episode reward + length)
        tb_logs/v2x_ppo_1/      ← TensorBoard event files
        eval_summary.csv        ← Post-training: 50-ep RL policy evaluation
        baseline_summary.csv    ← 50-ep V2I-only rule-based baseline

IMPORTANT — Single TraCI Connection:
    SUMO only allows ONE active TraCI connection per process. This script uses
    a single V2XEnv for training, closes it, then creates fresh envs for
    evaluation and baseline. DO NOT use EvalCallback with a separate eval_env
    — it will crash with "Connection refused" when it tries to reset mid-rollout.

Usage:
    python agent/train.py                        # full 500k-step run
    python agent/train.py --timesteps 5000       # quick smoke-test
    python agent/train.py --timesteps 500000 --eval-freq 50000
"""

import os
import sys
import argparse
import csv
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "env"))

from v2x_env import V2XEnv

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# ── Directory constants ───────────────────────────────────────────────────────
AGENT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(AGENT_DIR, "models")
LOGS_DIR    = os.path.join(AGENT_DIR, "logs")
TB_LOGS_DIR = os.path.join(LOGS_DIR, "tb_logs")

TRAIN_MONITOR_DIR    = os.path.join(LOGS_DIR, "training_monitor")
EVAL_SUMMARY_CSV     = os.path.join(LOGS_DIR, "eval_summary.csv")
BASELINE_SUMMARY_CSV = os.path.join(LOGS_DIR, "baseline_summary.csv")
FINAL_MODEL_PATH     = os.path.join(MODELS_DIR, "final_model")


def make_dirs():
    for d in [MODELS_DIR, LOGS_DIR, TB_LOGS_DIR, TRAIN_MONITOR_DIR]:
        os.makedirs(d, exist_ok=True)


def make_env(monitor_dir, tag="train"):
    """
    Factory returning a callable that creates a monitored V2XEnv.
    Monitor writes per-episode reward + length to a CSV for plotting.
    """
    def _init():
        env = V2XEnv()
        env = Monitor(env, filename=os.path.join(monitor_dir, tag))
        return env
    return _init


# ── Custom callback: log PDR + latency to TensorBoard ────────────────────────
class V2XMetricsCallback(BaseCallback):
    """
    Reads the 'pdr' and 'mean_latency' keys from the info dict returned by
    env.step() and logs them as TensorBoard scalars every step.
    These supplement SB3's built-in reward/ep_len metrics.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "pdr" in info:
                self.logger.record("v2x/pdr", info["pdr"])
            if "mean_latency" in info:
                self.logger.record("v2x/mean_latency_ms", info["mean_latency"])
        return True  # returning False would stop training


def train_ppo(total_timesteps: int, checkpoint_freq: int):
    """
    Full PPO training loop with a single SUMO environment.

    Architecture notes:
        - Policy: MlpPolicy — the (20, 4) observation is auto-flattened to 80-dim.
        - No EvalCallback: SUMO only allows one TraCI connection at a time.
          We save checkpoints via CheckpointCallback and evaluate separately
          after training completes.
        - V2XMetricsCallback: streams PDR + latency to TensorBoard.
    """
    print("=" * 60)
    print("  V2X Intelligent Communication — PPO Training")
    print(f"  Total timesteps    : {total_timesteps:,}")
    print(f"  Checkpoint every   : {checkpoint_freq:,} steps")
    print("=" * 60)

    make_dirs()

    # ── Single training environment ────────────────────────────────────────
    train_env = DummyVecEnv([make_env(TRAIN_MONITOR_DIR, tag="train")])

    # ── Callbacks ─────────────────────────────────────────────────────────
    checkpoint_callback = CheckpointCallback(
        save_freq   = checkpoint_freq,
        save_path   = MODELS_DIR,
        name_prefix = "v2x_ppo_checkpoint",
        verbose     = 1,
    )
    metrics_callback = V2XMetricsCallback(verbose=0)

    # ── PPO model ─────────────────────────────────────────────────────────
    model = PPO(
        policy        = "MlpPolicy",
        env           = train_env,
        learning_rate = 3e-4,
        n_steps       = 2048,       # Steps per rollout buffer
        batch_size    = 64,
        n_epochs      = 10,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        clip_range    = 0.2,
        ent_coef      = 0.01,       # Entropy bonus (encourages exploration)
        vf_coef       = 0.5,
        max_grad_norm = 0.5,
        policy_kwargs = {"net_arch": [256, 256]},
        tensorboard_log = TB_LOGS_DIR,
        verbose       = 1,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps = total_timesteps,
        callback        = [checkpoint_callback, metrics_callback],
        tb_log_name     = "v2x_ppo",
        reset_num_timesteps = True,
    )

    # Save the final weights regardless of checkpoint history
    model.save(FINAL_MODEL_PATH)
    print(f"\n[✓] Final model saved → {FINAL_MODEL_PATH}.zip")

    # IMPORTANT: close training env before opening a new SUMO connection
    train_env.close()
    print("[✓] Training environment closed (SUMO connection released)")

    return model


def _best_model_path():
    """
    Find the highest-step checkpoint saved by CheckpointCallback.
    Falls back to the final model if no checkpoints exist.
    """
    checkpoints = [
        f for f in os.listdir(MODELS_DIR)
        if f.startswith("v2x_ppo_checkpoint_") and f.endswith(".zip")
    ]
    if checkpoints:
        # Filenames: v2x_ppo_checkpoint_<steps>_steps.zip
        # Sort by the numeric steps field
        checkpoints.sort(key=lambda f: int(f.split("_")[3]))
        best = os.path.join(MODELS_DIR, checkpoints[-1])
        print(f"[→] Using latest checkpoint: {checkpoints[-1]}")
        return best
    return FINAL_MODEL_PATH + ".zip"


def run_evaluation(model, n_episodes=50, label="RL_PPO"):
    """
    Evaluate a trained model for `n_episodes` episodes.
    Opens a fresh V2XEnv (new SUMO process) — call only after train_env.close().

    Each row in the output CSV captures:
        episode, total_reward, mean_pdr, mean_latency_ms, v2v_ratio, v2i_ratio
    """
    print(f"\n[→] Running {n_episodes}-episode evaluation: {label}")
    rows = []
    env  = V2XEnv()

    for ep in range(n_episodes):
        obs, _     = env.reset()
        ep_reward  = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
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
            avg_r = np.mean([r["total_reward"] for r in rows[-10:]])
            avg_pdr = np.mean([r["mean_pdr"] for r in rows[-10:]])
            avg_lat = np.mean([r["mean_latency_ms"] for r in rows[-10:]])
            print(f"  Ep {ep+1:3d}/{n_episodes} | "
                  f"avg_reward={avg_r:.3f} | "
                  f"avg_PDR={avg_pdr:.3f} | "
                  f"avg_latency={avg_lat:.1f}ms")

    env.close()

    outfile = EVAL_SUMMARY_CSV if label == "RL_PPO" else BASELINE_SUMMARY_CSV
    _write_csv(rows, outfile)
    print(f"  [✓] Saved → {outfile}")
    return rows


def run_baseline(n_episodes=50):
    """
    Rule-Based Baseline: always V2I (action=0) for every vehicle slot.

    Represents the conventional infrastructure-centric approach — every
    vehicle routes through the base station with 99% PDR but high latency.
    This is the comparison target for the RL agent's latency improvements.
    Opens its own fresh V2XEnv — call only after all previous envs are closed.
    """
    print(f"\n[→] Running {n_episodes}-episode V2I-only baseline")
    rows = []
    env  = V2XEnv()

    for ep in range(n_episodes):
        obs, _     = env.reset()
        ep_reward  = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
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
            avg_r = np.mean([r["total_reward"] for r in rows[-10:]])
            print(f"  Ep {ep+1:3d}/{n_episodes} | avg_reward={avg_r:.3f}")

    env.close()
    _write_csv(rows, BASELINE_SUMMARY_CSV)
    print(f"  [✓] Baseline saved → {BASELINE_SUMMARY_CSV}")
    return rows


def _write_csv(rows: list, path: str):
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
        "--checkpoint-freq", type=int, default=50_000,
        help="Save a model checkpoint every N steps (default: 50,000)"
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=50,
        help="Number of post-training evaluation episodes (default: 50)"
    )
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="Skip the V2I-only baseline run (useful for quick smoke-tests)"
    )
    args = parser.parse_args()

    # ── Step 1: Train ──────────────────────────────────────────────────────
    # train_env is closed inside train_ppo() before returning
    model = train_ppo(args.timesteps, args.checkpoint_freq)

    # ── Step 2: Load best checkpoint for evaluation ───────────────────────
    best_path = _best_model_path()
    if os.path.exists(best_path):
        print(f"\n[→] Loading model from {best_path}")
        model = PPO.load(best_path)
    else:
        print("[!] No checkpoint found — using in-memory final model")

    # ── Step 3: Post-training RL Evaluation ───────────────────────────────
    # Fresh SUMO process — safe because train_env was closed in step 1
    run_evaluation(model, n_episodes=args.eval_episodes, label="RL_PPO")

    # ── Step 4: Baseline Comparison ───────────────────────────────────────
    # Another fresh SUMO process — safe because eval env was closed above
    if not args.skip_baseline:
        run_baseline(n_episodes=args.eval_episodes)

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Checkpoints       : {MODELS_DIR}/")
    print(f"  Training monitor  : {TRAIN_MONITOR_DIR}/train.monitor.csv")
    print(f"  RL eval results   : {EVAL_SUMMARY_CSV}")
    if not args.skip_baseline:
        print(f"  Baseline results  : {BASELINE_SUMMARY_CSV}")
    print("  TensorBoard       : tensorboard --logdir " + TB_LOGS_DIR)
    print("  → Run: python src/visualization/plot_results.py")
    print("=" * 60)


if __name__ == "__main__":
    main()