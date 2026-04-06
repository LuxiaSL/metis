#!/usr/bin/env python3
"""NCA checkpoint analysis: slope ratios, reversal detection, knee identification.

Based on luxia-base research methodology:
1. Compute slope ratios at varied window sizes to assess saturation
2. Detect metric reversals (direction flips that signal geometric instability)
3. Identify pre-reversal checkpoint candidates (optimal for downstream transfer)

Usage:
    # Analyze latest resume checkpoint
    python scripts/analyze_nca.py checkpoints/nca_phase1/nca_resume.pt

    # Compare multiple checkpoints
    python scripts/analyze_nca.py checkpoints/nca_phase1/nca_step_*.pt

    # Custom window sizes
    python scripts/analyze_nca.py checkpoints/nca_phase1/nca_resume.pt --windows 1000,2000,5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
import numpy as np


# ── Metric groups ────────────────────────────────────────────────────────

CORE_METRICS = ["loss", "eval_loss", "geo/rankme_last"]

GRADIENT_METRICS = ["grad_norm", "model_grad_norm", "head_grad_norm"]

# These get discovered dynamically from the checkpoint
LAYER_METRIC_SUFFIXES = [
    "anisotropy", "attn_entropy_mean", "attn_entropy_std",
    "stable_rank_q_proj", "stable_rank_o_proj",
    "stable_rank_gate_proj", "stable_rank_down_proj",
    "dead_units",
]


def load_checkpoint(path: str) -> dict:
    """Load checkpoint and extract metric_history + steps."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    mh = ckpt.get("metric_history", {})
    steps = mh.get("geo/step", [])
    if steps:
        steps = [int(s) for s in steps]
    return {
        "path": path,
        "step": ckpt.get("step", "?"),
        "loss": ckpt.get("loss", "?"),
        "best_loss": ckpt.get("best_loss", "?"),
        "seed": ckpt.get("seed", "?"),
        "metric_history": mh,
        "steps": steps,
    }


# ── Slope ratio analysis ────────────────────────────────────────────────


def compute_slope_ratio(
    values: list[float],
    steps: list[float],
    window_start: int,
    window_end: int,
) -> Optional[tuple[float, float, float]]:
    """Compute slope ratio for a metric over a step range.

    Splits the range into first half and second half, computes absolute
    change in each, returns (early_change, late_change, ratio).

    Returns None if insufficient data in the range.
    """
    # Find indices within the step range, clamped to values length
    n_vals = len(values)
    indices = [i for i, s in enumerate(steps) if window_start <= s <= window_end and i < n_vals]
    if len(indices) < 4:
        return None

    vals = [values[i] for i in indices]
    mid = len(vals) // 2

    early_change = abs(vals[mid] - vals[0])
    late_change = abs(vals[-1] - vals[mid])
    early_change = max(early_change, 1e-10)
    ratio = late_change / early_change

    return early_change, late_change, ratio


def slope_analysis(
    mh: dict[str, list[float]],
    steps: list[float],
    windows: list[tuple[int, int]],
    metrics: Optional[list[str]] = None,
) -> dict[str, list[dict]]:
    """Compute slope ratios for all metrics across multiple windows.

    Returns dict mapping metric name to list of window results.
    """
    if metrics is None:
        metrics = [k for k in mh if k not in ("geo/step", "geo/tier1_time_s")]

    results: dict[str, list[dict]] = {}
    for key in sorted(metrics):
        if key not in mh or len(mh[key]) < 4:
            continue
        vals = mh[key]
        window_results = []
        for w_start, w_end in windows:
            sr = compute_slope_ratio(vals, steps, w_start, w_end)
            if sr is not None:
                early, late, ratio = sr
                status = "PLATEAUED" if ratio < 0.3 else "SLOWING" if ratio < 0.5 else "ACTIVE"
                window_results.append({
                    "window": f"{w_start}-{w_end}",
                    "early_change": early,
                    "late_change": late,
                    "ratio": ratio,
                    "status": status,
                })
        if window_results:
            results[key] = window_results

    return results


# ── Reversal detection ───────────────────────────────────────────────────


def detect_reversals(
    values: list[float],
    steps: list[float],
    smooth_window: int = 3,
) -> list[dict]:
    """Detect direction reversals in a metric's trajectory.

    Uses a smoothing window to avoid noise-triggered false positives.
    Returns list of reversal events with step, direction, magnitude.
    """
    if len(values) < smooth_window * 2 + 1:
        return []

    # Align steps to values length (some metrics start later)
    steps = steps[:len(values)]

    # Smooth with rolling mean
    arr = np.array(values, dtype=np.float64)
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(arr, kernel, mode="valid")
        # Adjust step indices to match smoothed array
        offset = smooth_window // 2
        smooth_steps = steps[offset:offset + len(smoothed)]
    else:
        smoothed = arr
        smooth_steps = steps

    # Compute deltas
    deltas = np.diff(smoothed)
    delta_steps = smooth_steps[1:]

    # Find sign changes
    reversals = []
    for i in range(1, len(deltas)):
        if deltas[i - 1] * deltas[i] < 0:
            direction = "UP→DOWN" if deltas[i - 1] > 0 else "DOWN→UP"
            magnitude = abs(deltas[i] - deltas[i - 1])
            reversals.append({
                "step": int(delta_steps[i]) if i < len(delta_steps) else -1,
                "direction": direction,
                "magnitude": magnitude,
            })

    return reversals


def reversal_analysis(
    mh: dict[str, list[float]],
    steps: list[float],
    metrics: Optional[list[str]] = None,
    smooth_window: int = 3,
) -> dict[str, list[dict]]:
    """Detect reversals across all metrics."""
    if metrics is None:
        metrics = [k for k in mh if k not in ("geo/step", "geo/tier1_time_s")]

    results: dict[str, list[dict]] = {}
    for key in sorted(metrics):
        if key not in mh or len(mh[key]) < 6:
            continue
        revs = detect_reversals(mh[key], steps, smooth_window)
        if revs:
            results[key] = revs

    return results


# ── Knee identification ──────────────────────────────────────────────────


def find_knee_candidates(
    mh: dict[str, list[float]],
    steps: list[float],
    reversal_results: dict[str, list[dict]],
) -> list[dict]:
    """Identify checkpoint candidates based on reversal onset.

    The optimal checkpoint is just BEFORE the first cluster of reversals.
    Returns candidates sorted by score (higher = better).
    """
    if not steps:
        return []

    # Count reversals per step bucket (1000-step buckets)
    max_step = max(steps)
    bucket_size = max(1000, max_step // 20)
    buckets: dict[int, int] = {}

    for key, revs in reversal_results.items():
        # Weight layer metrics higher than scalar metrics
        weight = 2.0 if "layer_" in key else 1.0
        for rev in revs:
            bucket = (rev["step"] // bucket_size) * bucket_size
            buckets[bucket] = buckets.get(bucket, 0) + weight

    if not buckets:
        return [{"step": max_step, "reason": "no reversals detected", "score": 0}]

    # Find the first bucket with significant reversals
    sorted_buckets = sorted(buckets.items())
    reversal_onset = None
    for bucket_step, count in sorted_buckets:
        if count >= 3:  # At least 3 weighted reversals in a bucket
            reversal_onset = bucket_step
            break

    if reversal_onset is None:
        reversal_onset = sorted_buckets[-1][0]

    # Find checkpoints just before reversal onset
    available_steps = sorted(set(
        s for s in range(0, max_step + 1, 1000) if s <= reversal_onset
    ))

    candidates = []
    for s in available_steps[-3:]:  # Last 3 checkpoints before onset
        # Score: prefer later (more training) but before reversals
        dist_to_reversal = reversal_onset - s
        score = s / max_step - (dist_to_reversal / max_step) * 0.1
        candidates.append({
            "step": s,
            "reason": f"pre-reversal (onset ~{reversal_onset})",
            "score": score,
        })

    # Also include eval_loss minimum if available
    if "eval_loss" in mh:
        eval_vals = mh["eval_loss"]
        min_idx = int(np.argmin(eval_vals))
        if min_idx < len(steps):
            candidates.append({
                "step": int(steps[min_idx]),
                "reason": f"eval_loss minimum ({eval_vals[min_idx]:.4f})",
                "score": 0.5,
            })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


# ── Display ──────────────────────────────────────────────────────────────


def print_summary(ckpt: dict) -> None:
    """Print checkpoint overview."""
    mh = ckpt["metric_history"]
    steps = ckpt["steps"]

    print("=" * 80)
    print(f"NCA CHECKPOINT ANALYSIS: {ckpt['path']}")
    print(f"  Step: {ckpt['step']}, Loss: {ckpt['loss']}, Best: {ckpt['best_loss']}")
    if steps:
        print(f"  Data range: step {min(steps)} - {max(steps)}, {len(steps)} measurements")
        # Detect frequency changes
        diffs = [steps[i + 1] - steps[i] for i in range(len(steps) - 1)]
        unique_diffs = sorted(set(diffs))
        print(f"  Sampling intervals: {unique_diffs}")
    print(f"  Metrics tracked: {len(mh)} keys")
    grad_keys = [k for k in mh if "grad" in k]
    if grad_keys:
        print(f"  Gradient metrics: {grad_keys}")
    else:
        print("  Gradient metrics: (none)")
    print("=" * 80)


def print_raw_values(mh: dict, steps: list, metrics: list[str]) -> None:
    """Print raw metric values at each step."""
    print("\nRAW VALUES (last 10 measurements):")
    available = [k for k in metrics if k in mh and mh[k]]
    if not available:
        print("  (no data)")
        return

    n = min(10, len(steps))
    show_steps = steps[-n:]
    header = f"{'Metric':<50s}" + "".join(f" | {s:>7}" for s in show_steps)
    print(header)
    print("-" * len(header))

    for key in available:
        vals = mh[key][-n:]
        row = f"{key:<50s}"
        for v in vals:
            if abs(v) < 0.01:
                row += f" | {v:>7.5f}"
            elif abs(v) < 100:
                row += f" | {v:>7.3f}"
            else:
                row += f" | {v:>7.1f}"
        print(row)


def print_slope_analysis(results: dict[str, list[dict]]) -> None:
    """Print slope ratio table."""
    print("\nSLOPE RATIOS:")
    if not results:
        print("  (insufficient data)")
        return

    # Get all windows
    all_windows = set()
    for window_results in results.values():
        for wr in window_results:
            all_windows.add(wr["window"])
    all_windows = sorted(all_windows)

    header = f"{'Metric':<50s}" + "".join(f" | {w:>15s}" for w in all_windows)
    print(header)
    print("-" * len(header))

    for key in sorted(results.keys()):
        row = f"{key:<50s}"
        window_map = {wr["window"]: wr for wr in results[key]}
        for w in all_windows:
            if w in window_map:
                wr = window_map[w]
                row += f" | {wr['ratio']:>5.3f} {wr['status']:>8s}"
            else:
                row += f" | {'':>15s}"
        print(row)


def print_reversals(results: dict[str, list[dict]]) -> None:
    """Print reversal detection results."""
    print("\nREVERSAL DETECTION:")
    if not results:
        print("  No reversals detected")
        return

    total = sum(len(revs) for revs in results.values())
    print(f"  {total} reversals across {len(results)} metrics")
    print()

    for key in sorted(results.keys()):
        revs = results[key]
        rev_strs = [f"step {r['step']} ({r['direction']})" for r in revs]
        print(f"  {key}:")
        for rs in rev_strs:
            print(f"    {rs}")


def print_candidates(candidates: list[dict]) -> None:
    """Print knee candidates."""
    print("\nKNEE CANDIDATES (pre-reversal checkpoints):")
    if not candidates:
        print("  (no candidates)")
        return

    for i, c in enumerate(candidates):
        marker = ">>> " if i == 0 else "    "
        print(f"  {marker}step {c['step']:>6d}  score={c['score']:.3f}  {c['reason']}")


# ── Main ─────────────────────────────────────────────────────────────────


def analyze_checkpoint(path: str, window_sizes: list[int], smooth: int) -> None:
    """Full analysis of a single checkpoint."""
    ckpt = load_checkpoint(path)
    mh = ckpt["metric_history"]
    steps = ckpt["steps"]

    if not steps:
        print(f"No step data in {path}")
        return

    print_summary(ckpt)

    # Discover all metrics
    layer_metrics = [k for k in mh if k.startswith("geo/layer_")]
    all_metrics = CORE_METRICS + GRADIENT_METRICS + sorted(layer_metrics)
    available = [k for k in all_metrics if k in mh]

    # Raw values
    print_raw_values(mh, steps, available)

    # Build windows from the data range
    max_step = max(steps)
    windows = []
    for ws in window_sizes:
        if ws < max_step:
            windows.append((0, ws))
            windows.append((max_step - ws, max_step))
    # Full range
    windows.append((0, max_step))
    windows = sorted(set(windows))

    # Slope analysis
    slope_results = slope_analysis(mh, steps, windows, available)
    print_slope_analysis(slope_results)

    # Reversal detection
    rev_results = reversal_analysis(mh, steps, available, smooth)
    print_reversals(rev_results)

    # Knee candidates
    candidates = find_knee_candidates(mh, steps, rev_results)
    print_candidates(candidates)

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="NCA checkpoint knee analysis")
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint .pt files to analyze")
    parser.add_argument("--windows", type=str, default="2000,5000,10000",
                        help="Window sizes for slope ratio (comma-separated)")
    parser.add_argument("--smooth", type=int, default=3,
                        help="Smoothing window for reversal detection")
    args = parser.parse_args()

    window_sizes = [int(w) for w in args.windows.split(",")]

    for path in args.checkpoints:
        if not Path(path).exists():
            print(f"Not found: {path}")
            continue
        analyze_checkpoint(path, window_sizes, args.smooth)


if __name__ == "__main__":
    main()
