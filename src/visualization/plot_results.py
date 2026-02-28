"""
V2X Research Paper Visualization Suite — STRICT REAL-DATA MODE
===============================================================
Academic-grade plotting script for the Intelligent V2X Communication System.

*** SYNTHETIC DATA POLICY ***
This script operates under a strict no-mock-data policy required for academic
publication. It will NEVER generate, simulate, or hallucinate training metrics.

If any required input CSV is absent, the script raises FileNotFoundError
immediately and terminates. You MUST run agent/train.py first.

Required Input Files (all produced by agent/train.py):
    agent/logs/training_monitor/train.monitor.csv
    agent/logs/eval_summary.csv
    agent/logs/baseline_summary.csv

Output Files (written to v2x_results/figures/):
    fig1_learning_curve.png
    fig2_latency_comparison.png
    fig3_pdr_throughput.png
    fig4_mode_distribution.png

Usage:
    cd /Users/vinayakprakash/Documents/V2X-COMMUNICATION
    MPLBACKEND=Agg python3 src/visualization/plot_results.py

Exporting Telemetry from RC Car / SUMO:
    Each row of eval_summary.csv needs:
        episode, total_reward, mean_pdr, mean_latency_ms, v2v_ratio, v2i_ratio
    Log raw CAN-bus fields [timestamp_ms, vehicle_id, mode, latency_ms, packet_ok]
    per packet, then aggregate per session to match this schema.
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ─── Path constants ───────────────────────────────────────────────────────────
PROJECT_ROOT     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MONITOR  = os.path.join(PROJECT_ROOT, "agent", "logs", "training_monitor", "train.monitor.csv")
DEFAULT_EVAL     = os.path.join(PROJECT_ROOT, "agent", "logs", "eval_summary.csv")
DEFAULT_BASELINE = os.path.join(PROJECT_ROOT, "agent", "logs", "baseline_summary.csv")
DEFAULT_OUTDIR   = os.path.join(PROJECT_ROOT, "v2x_results", "figures")

# ─── Physics constants (must match v2x_env.py exactly) ───────────────────────
# Any deviation from these bounds in real data indicates a logging bug.
V2I_LATENCY_EXPECTED_MS    = 20.0   # N(20, 5) — baseline V2I mean
V2I_LATENCY_TOLERANCE_MS   = 8.0    # ±8ms accepted range around 20ms mean
V2V_LATENCY_MAX_MS         = 10.0   # V2V must always be below this (direct link)
PDR_MIN                    = 0.0
PDR_MAX                    = 1.0

# ─── Academic Design System ───────────────────────────────────────────────────
C_RL       = "#1565C0"   # Deep blue  — RL PPO agent
C_BASE     = "#D84315"   # Deep orange — Baseline (V2I-only)
C_V2V      = "#2E7D32"   # Forest green — V2V mode
C_V2I      = "#F9A825"   # Amber        — V2I mode
C_FILL_RL  = "#90CAF9"
C_FILL_B   = "#FFAB91"

DPI        = 300
FIG_W      = 8
FIG_H      = 5
ROLLING_WIN = 10


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — HARD FILE VALIDATION
# Raises FileNotFoundError immediately if any required CSV is missing.
# This is the first thing that runs; no data is loaded before this passes.
# ══════════════════════════════════════════════════════════════════════════════

def require_file(path: str, label: str) -> None:
    """
    Strictly assert that a required data file exists.
    Raises FileNotFoundError with a clear academic context message on failure.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n\n{'='*70}\n"
            f"  MISSING REQUIRED DATA FILE\n"
            f"{'='*70}\n"
            f"  File   : {path}\n"
            f"  Label  : {label}\n\n"
            f"  ERROR  : Real training/evaluation data not found.\n"
            f"           You must run train.py with the SUMO simulation first.\n\n"
            f"  Run    : python3 agent/train.py --timesteps 500000\n"
            f"{'='*70}\n"
        )


def validate_all_inputs(monitor_path: str, eval_path: str, baseline_path: str) -> None:
    """
    Gate function — validates the existence of ALL required input files before
    any plotting or loading logic is executed.
    """
    print("\n[VALIDATION] Checking required input files...")
    require_file(monitor_path,  "Training Monitor CSV  (agent/logs/training_monitor/train.monitor.csv)")
    require_file(eval_path,     "RL Eval Summary CSV   (agent/logs/eval_summary.csv)")
    require_file(baseline_path, "Baseline Summary CSV  (agent/logs/baseline_summary.csv)")
    print("[VALIDATION] All required files found. ✓\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_monitor_csv(path: str) -> pd.DataFrame:
    """
    Parse SB3 Monitor CSV. Format:
        Line 0: JSON comment  {"t_start": ..., "env_id": ...}
        Line 1: header        r,l,t
    """
    df = pd.read_csv(path, skiprows=1, header=0)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"r": "reward", "l": "ep_length", "t": "elapsed_s"})
    df["episode"] = range(1, len(df) + 1)
    if df.empty:
        raise ValueError(
            f"Monitor CSV at {path} is empty. "
            "Training may not have completed any full episodes."
        )
    return df


def load_eval_csv(path: str, label: str) -> pd.DataFrame:
    """Load and basic-validate an evaluation summary CSV."""
    df = pd.read_csv(path)
    required_cols = {"episode", "total_reward", "mean_pdr", "mean_latency_ms",
                     "v2v_ratio", "v2i_ratio"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"[{label}] CSV is missing required columns: {missing}\n"
            f"  File: {path}\n"
            "  Ensure train.py wrote all telemetry fields correctly."
        )
    if df.empty:
        raise ValueError(f"[{label}] CSV at {path} contains no rows.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — PHYSICS / ACADEMIC SANITY CHECKS
# Verifies that loaded data is consistent with the physics model defined in
# v2x_env.py. A wild deviation indicates a logging bug or data corruption.
# ══════════════════════════════════════════════════════════════════════════════

def _check_range(value: float, lo: float, hi: float, label: str) -> None:
    if not (lo <= value <= hi):
        raise ValueError(
            f"\n[SANITY CHECK FAILED] {label}\n"
            f"  Expected range : [{lo:.2f}, {hi:.2f}]\n"
            f"  Actual value   : {value:.4f}\n"
            f"  This violates the physics model defined in v2x_env.py.\n"
            f"  Possible causes: logging bug, unit mismatch, or data corruption."
        )


def run_physics_sanity_checks(eval_df: pd.DataFrame, baseline_df: pd.DataFrame) -> None:
    """
    Validate that the loaded DataFrames obey the physics rules of v2x_env.py.

    Check 1 — V2I Baseline Latency
        The baseline always uses V2I (mode=0). v2x_env.py samples latency from
        N(20ms, 5ms). The empirical mean across 50 episodes must sit within
        V2I_LATENCY_EXPECTED_MS ± V2I_LATENCY_TOLERANCE_MS.

    Check 2 — V2V Latency Upper Bound
        V2V latency is sampled from N(2ms, 0.5ms). Even with outliers, the
        episode mean must never exceed V2V_LATENCY_MAX_MS (10ms) in RL episodes.

    Check 3 — PDR Bounds
        PDR must be in [0, 1] for every row in both datasets.

    Check 4 — V2V Interference Rule
        In RL episodes where v2v_ratio > 0.5 AND mean_pdr < 0.80, the PDR
        depression is consistent with the interference model (>5 neighbours
        causes drops). We verify this relationship holds directionally.

    Check 5 — Baseline Mode Purity
        The baseline policy always uses V2I (action=0). v2i_ratio must be
        exactly 1.0 in every baseline row.
    """
    print("[SANITY CHECK] Running physics-based data validation...")

    # ── Check 1: V2I latency window ──────────────────────────────────────────
    baseline_lat_mean = baseline_df["mean_latency_ms"].mean()
    lo = V2I_LATENCY_EXPECTED_MS - V2I_LATENCY_TOLERANCE_MS
    hi = V2I_LATENCY_EXPECTED_MS + V2I_LATENCY_TOLERANCE_MS
    _check_range(
        baseline_lat_mean, lo, hi,
        f"Baseline V2I latency mean ({baseline_lat_mean:.2f} ms) must be "
        f"~{V2I_LATENCY_EXPECTED_MS}ms ± {V2I_LATENCY_TOLERANCE_MS}ms"
    )
    print(f"  [✓] Check 1 — Baseline V2I latency mean: {baseline_lat_mean:.2f} ms  (expected ~20 ms)")

    # ── Check 2: V2V latency upper bound ─────────────────────────────────────
    rl_lat_mean = eval_df["mean_latency_ms"].mean()
    _check_range(
        rl_lat_mean, 0.0, V2V_LATENCY_MAX_MS,
        f"RL episode mean latency ({rl_lat_mean:.2f} ms) must be < {V2V_LATENCY_MAX_MS} ms "
        f"when V2V dominates (V2V mean << 10ms, V2I mean ~20ms → mix must be < 10ms)"
    )
    print(f"  [✓] Check 2 — RL mean latency: {rl_lat_mean:.2f} ms  (must be < {V2V_LATENCY_MAX_MS} ms)")

    # ── Check 3: PDR must be in [0, 1] ───────────────────────────────────────
    for label, df in [("RL eval", eval_df), ("Baseline", baseline_df)]:
        pdr_min = df["mean_pdr"].min()
        pdr_max = df["mean_pdr"].max()
        _check_range(pdr_min, PDR_MIN, PDR_MAX, f"{label} PDR min ({pdr_min:.4f})")
        _check_range(pdr_max, PDR_MIN, PDR_MAX, f"{label} PDR max ({pdr_max:.4f})")
    print(f"  [✓] Check 3 — PDR values in [0, 1] for both datasets")

    # ── Check 4: V2V interference rule (directional) ─────────────────────────
    # When agent heavily uses V2V (> 50%), some PDR drop from interference
    # should manifest as episodes with pdr < 0.95. This validates the physics.
    high_v2v_eps = eval_df[eval_df["v2v_ratio"] > 0.5]
    if len(high_v2v_eps) > 0:
        pdr_in_high_v2v = high_v2v_eps["mean_pdr"].mean()
        # If V2V is used heavily and PDR is suspiciously perfect (1.0 always),
        # that would mean interference was never triggered — a physics violation.
        if pdr_in_high_v2v == 1.0 and len(high_v2v_eps) > 10:
            raise ValueError(
                f"\n[SANITY CHECK FAILED] V2V Interference Rule\n"
                f"  {len(high_v2v_eps)} episodes have v2v_ratio > 0.5.\n"
                f"  But mean PDR in those episodes is exactly 1.0.\n"
                f"  This is physically impossible — the interference model in\n"
                f"  v2x_env.py must cause some packet drops at high V2V density.\n"
                f"  Likely cause: logging bug — PDR is not being recorded correctly."
            )
        print(f"  [✓] Check 4 — V2V interference rule: mean PDR in high-V2V eps = {pdr_in_high_v2v:.4f}")
    else:
        print("  [~] Check 4 — Skipped: no episodes with v2v_ratio > 0.5 found")

    # ── Check 5: Baseline mode purity ─────────────────────────────────────────
    non_pure_v2i = baseline_df[baseline_df["v2i_ratio"] < 0.999]
    if len(non_pure_v2i) > 0:
        raise ValueError(
            f"\n[SANITY CHECK FAILED] Baseline Mode Purity\n"
            f"  {len(non_pure_v2i)} baseline episode(s) have v2i_ratio < 1.0.\n"
            f"  The baseline policy always uses V2I (action=0); any V2V usage\n"
            f"  indicates a bug in run_baseline() inside agent/train.py.\n"
            f"  Affected rows:\n{non_pure_v2i[['episode','v2i_ratio']].to_string()}"
        )
    print(f"  [✓] Check 5 — Baseline V2I purity: all {len(baseline_df)} episodes use 100% V2I")

    print("[SANITY CHECK] All physics checks passed. ✓\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — PLOTTING (real data only)
# ══════════════════════════════════════════════════════════════════════════════

def _apply_global_style():
    matplotlib.rcParams.update({
        "font.family":        "DejaVu Serif",
        "font.size":          12,
        "axes.titlesize":     14,
        "axes.labelsize":     12,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.fontsize":    10,
        "legend.framealpha":  0.9,
        "figure.dpi":         DPI,
        "savefig.dpi":        DPI,
        "savefig.bbox":       "tight",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "text.usetex":        False,
        "mathtext.fontset":   "dejavuserif",
    })
    sns.set_theme(style="whitegrid", font_scale=1.1)


def _save(fig, outdir: str, filename: str):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"  [✓] Saved → {path}")
    plt.close(fig)


# ─── Figure 1: Learning Curve ─────────────────────────────────────────────────
def plot_learning_curve(df: pd.DataFrame, outdir: str):
    eps       = df["episode"].values
    rews      = df["reward"].values
    s         = pd.Series(rews)
    roll_mean = s.rolling(ROLLING_WIN, min_periods=1).mean().values
    roll_std  = s.rolling(ROLLING_WIN, min_periods=1).std(ddof=0).fillna(0).values

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(eps, rews,      color=C_FILL_RL, linewidth=0.6, alpha=0.55, zorder=1)
    ax.plot(eps, roll_mean, color=C_RL,      linewidth=2.0,
            label=f"Rolling mean (window={ROLLING_WIN})", zorder=3)
    ax.fill_between(
        eps,
        roll_mean - roll_std,
        roll_mean + roll_std,
        color=C_FILL_RL, alpha=0.35,
        label=r"$\pm 1\,\sigma$ (rolling std)", zorder=2,
    )

    # Annotate total episodes and final mean reward
    final_mean = roll_mean[-1]
    ax.axhline(final_mean, color=C_RL, linestyle=":", linewidth=1.0, alpha=0.6)
    ax.text(eps[-1] * 0.02, final_mean + 0.05,
            f"Final mean: {final_mean:.3f}", color=C_RL, fontsize=9)

    # Convergence region (last 15% of training)
    conv_start = int(len(eps) * 0.85)
    ax.axvspan(eps[conv_start], eps[-1], color="#E8F5E9", alpha=0.4)
    ax.annotate(
        "Convergence",
        xy=(eps[conv_start + (len(eps) - conv_start) // 2], roll_mean[conv_start]),
        fontsize=9, color="#2E7D32", ha="center",
    )

    ax.set_xlabel("Training Episode")
    ax.set_ylabel(r"Cumulative Reward $\mathit{R}$")
    ax.set_title(f"Fig. 1 — PPO Learning Curve: V2X Mode Selection  "
                 f"[{len(eps)} episodes, real SUMO data]")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    _save(fig, outdir, "fig1_learning_curve.png")


# ─── Figure 2: Latency Comparison ────────────────────────────────────────────
def plot_latency_comparison(rl_df: pd.DataFrame, base_df: pd.DataFrame, outdir: str):
    metrics = {
        "policy":   ["RL-PPO (Adaptive)", "Baseline (V2I Only)"],
        "lat_mean": [rl_df["mean_latency_ms"].mean(),  base_df["mean_latency_ms"].mean()],
        "lat_std":  [rl_df["mean_latency_ms"].std(),   base_df["mean_latency_ms"].std()],
    }
    x      = np.arange(len(metrics["policy"]))
    colors = [C_RL, C_BASE]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        x, metrics["lat_mean"],
        yerr=metrics["lat_std"], capsize=6,
        color=colors, edgecolor="black", linewidth=0.8, width=0.45,
        error_kw={"elinewidth": 1.5, "ecolor": "black", "capthick": 1.5},
    )
    for bar, val, err in zip(bars, metrics["lat_mean"], metrics["lat_std"]):
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + err + 0.3,
                f"{val:.1f} ms", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics["policy"], fontsize=11)
    ax.set_ylabel(r"Mean End-to-End Latency (ms)")
    ax.set_title(f"Fig. 2 — Average Communication Latency: RL vs. Baseline  "
                 f"[{len(rl_df)} eval episodes, real SUMO data]")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, max(metrics["lat_mean"]) * 1.4)
    legend_patches = [
        mpatches.Patch(color=C_RL,   label="RL-PPO (Adaptive V2V/V2I)"),
        mpatches.Patch(color=C_BASE, label="Baseline (V2I Only)"),
    ]
    ax.legend(handles=legend_patches, loc="upper right")
    fig.tight_layout()
    _save(fig, outdir, "fig2_latency_comparison.png")


# ─── Figure 3: PDR / Throughput ───────────────────────────────────────────────
def plot_pdr_throughput(rl_df: pd.DataFrame, base_df: pd.DataFrame, outdir: str):
    rl_smooth   = pd.Series(rl_df["mean_pdr"].values).rolling(5, min_periods=1).mean()
    base_smooth = pd.Series(base_df["mean_pdr"].values).rolling(5, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(rl_df["episode"],   rl_df["mean_pdr"],    color=C_FILL_RL, linewidth=0.7, alpha=0.5)
    ax.plot(rl_df["episode"],   rl_smooth,             color=C_RL,     linewidth=2.2, label="RL-PPO (Adaptive)")
    ax.plot(base_df["episode"], base_df["mean_pdr"],   color=C_FILL_B, linewidth=0.7, alpha=0.5)
    ax.plot(base_df["episode"], base_smooth,           color=C_BASE,   linewidth=2.2,
            label="Baseline (V2I Only)", linestyle="--")
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=1, label="PDR = 1.0 (theoretical max)")

    common_ep = min(len(rl_df), len(base_df))
    ax.fill_between(
        rl_df["episode"].values[:common_ep],
        base_smooth.values[:common_ep],
        rl_smooth.values[:common_ep],
        where=rl_smooth.values[:common_ep] > base_smooth.values[:common_ep],
        color=C_V2V, alpha=0.15, label="PDR improvement",
    )

    ax.annotate(
        f"RL Mean: {rl_df['mean_pdr'].mean():.3f}",
        xy=(rl_df["episode"].iloc[-1], rl_smooth.iloc[-1]),
        xytext=(-90, 8), textcoords="offset points",
        fontsize=9, color=C_RL,
        arrowprops=dict(arrowstyle="->", color=C_RL, lw=1),
    )
    ax.annotate(
        f"Baseline Mean: {base_df['mean_pdr'].mean():.3f}",
        xy=(base_df["episode"].iloc[-1], base_smooth.iloc[-1]),
        xytext=(-110, -18), textcoords="offset points",
        fontsize=9, color=C_BASE,
        arrowprops=dict(arrowstyle="->", color=C_BASE, lw=1),
    )

    ax.set_xlabel("Evaluation Episode")
    ax.set_ylabel(r"Packet Delivery Ratio (PDR)")
    ax.set_title(f"Fig. 3 — Network Throughput: PDR over Evaluation Episodes  "
                 f"[real SUMO data]")
    ax.set_ylim(max(0.0, min(rl_df["mean_pdr"].min(), base_df["mean_pdr"].min()) - 0.05), 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    _save(fig, outdir, "fig3_pdr_throughput.png")


# ─── Figure 4: Mode Distribution ─────────────────────────────────────────────
def plot_mode_distribution(rl_df: pd.DataFrame, outdir: str):
    eps    = rl_df["episode"].values
    v2v_r  = rl_df["v2v_ratio"].values
    v2i_r  = rl_df["v2i_ratio"].values
    v2v_sm = pd.Series(v2v_r).rolling(5, min_periods=1).mean().values
    v2i_sm = pd.Series(v2i_r).rolling(5, min_periods=1).mean().values

    fig, (ax_stack, ax_pie) = plt.subplots(
        1, 2, figsize=(FIG_W + 3, FIG_H),
        gridspec_kw={"width_ratios": [3, 1.5]}
    )

    ax_stack.stackplot(
        eps, [v2v_sm, v2i_sm],
        labels=["V2V (Direct)", "V2I (Base Station)"],
        colors=[C_V2V, C_V2I], alpha=0.82,
    )
    ax_stack.set_xlabel("Evaluation Episode")
    ax_stack.set_ylabel("Mode Selection Ratio")
    ax_stack.set_title("Fig. 4a — Mode Distribution over Episodes")
    ax_stack.set_ylim(0, 1)
    ax_stack.legend(loc="upper right")
    ax_stack.grid(True, linestyle="--", alpha=0.4)

    mean_v2v = float(np.mean(v2v_r))
    mean_v2i = float(np.mean(v2i_r))
    ax_pie.pie(
        [mean_v2v, mean_v2i],
        labels=["V2V", "V2I"],
        colors=[C_V2V, C_V2I],
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
        textprops={"fontsize": 11}, pctdistance=0.7,
    )
    ax_pie.set_title("Fig. 4b — Overall\nMode Split", fontsize=11)

    fig.suptitle(
        f"Agent Mode Selection: V2V vs V2I Decision Distribution  "
        f"[{len(rl_df)} real episodes]",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    _save(fig, outdir, "fig4_mode_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="V2X Research Paper Figures — REAL DATA ONLY. "
                    "Run agent/train.py first."
    )
    parser.add_argument("--monitor",  default=DEFAULT_MONITOR,  help="Training monitor CSV")
    parser.add_argument("--eval",     default=DEFAULT_EVAL,     help="RL eval summary CSV")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE, help="Baseline summary CSV")
    parser.add_argument("--outdir",   default=DEFAULT_OUTDIR,   help="Output directory")
    parser.add_argument(
        "--skip-sanity", action="store_true",
        help="Skip physics sanity checks (not recommended for paper submission)"
    )
    args = parser.parse_args()

    _apply_global_style()

    print("=" * 70)
    print("  V2X Visualization Suite — STRICT REAL-DATA MODE")
    print("  No synthetic fallbacks. All figures from SUMO simulation logs.")
    print(f"  Output directory : {args.outdir}")
    print("=" * 70)

    # ── Step 1: Gate — fail fast if any CSV is missing ────────────────────────
    validate_all_inputs(args.monitor, args.eval, args.baseline)

    # ── Step 2: Load real data ─────────────────────────────────────────────────
    print("[LOADING] Reading CSV files from disk...")
    monitor_df  = load_monitor_csv(args.monitor)
    eval_df     = load_eval_csv(args.eval,     "RL Eval")
    baseline_df = load_eval_csv(args.baseline, "Baseline")

    print(f"  Training episodes : {len(monitor_df)}")
    print(f"  RL eval episodes  : {len(eval_df)}")
    print(f"  Baseline episodes : {len(baseline_df)}\n")

    # ── Step 3: Physics sanity checks ─────────────────────────────────────────
    if not args.skip_sanity:
        run_physics_sanity_checks(eval_df, baseline_df)
    else:
        print("[WARNING] Physics sanity checks SKIPPED. "
              "Not recommended for academic submission.\n")

    # ── Step 4: Generate all 4 figures ────────────────────────────────────────
    print("[PLOTTING] Generating publication figures from real data...")

    print("[→] Fig 1: Learning Curve")
    plot_learning_curve(monitor_df, args.outdir)

    print("[→] Fig 2: Latency Comparison")
    plot_latency_comparison(eval_df, baseline_df, args.outdir)

    print("[→] Fig 3: PDR / Throughput")
    plot_pdr_throughput(eval_df, baseline_df, args.outdir)

    print("[→] Fig 4: Mode Selection Distribution")
    plot_mode_distribution(eval_df, args.outdir)

    print("\n" + "=" * 70)
    print("  All 4 figures generated from 100% real SUMO simulation data.")
    print(f"  → Open : {args.outdir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
