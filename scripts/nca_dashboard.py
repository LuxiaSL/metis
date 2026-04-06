#!/usr/bin/env python3
"""NCA training dashboard — live snapshot of all metrics with epoch-relative analysis.

Usage:
    python scripts/nca_dashboard.py [checkpoint_path]

    Default: checkpoints/nca_phase1/nca_resume.pt
"""

import sys
from pathlib import Path

import torch
import numpy as np


DATASET_SIZE = 7_315_200  # train pairs
BATCH_SIZE = 256
STEPS_PER_EPOCH = DATASET_SIZE / BATCH_SIZE  # ~28,560


def load(path: str) -> tuple[dict, list[int], int]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    mh = ckpt["metric_history"]
    steps = [int(s) for s in mh.get("geo/step", [])]
    return mh, steps, ckpt.get("step", 0)


def trend(vals: list[float], steps: list[int]) -> dict:
    """Compute trend statistics for a metric series."""
    arr = np.array(vals, dtype=np.float64)
    x = np.arange(len(arr), dtype=np.float64)
    r = np.corrcoef(x, arr)[0, 1] if len(arr) > 2 and np.std(arr) > 1e-12 else 0.0
    slope = np.polyfit(x, arr, 1)[0] if len(arr) > 2 else 0.0
    return {
        "first": arr[0], "last": arr[-1], "min": arr.min(), "max": arr.max(),
        "mean": arr.mean(), "std": arr.std(),
        "r": r, "slope": slope,
        "direction": "UP" if slope > 0 else "DOWN",
        "strength": "strong" if abs(r) > 0.7 else "weak" if abs(r) > 0.3 else "noisy",
        "delta_pct": (arr[-1] - arr[0]) / abs(arr[0]) * 100 if abs(arr[0]) > 1e-10 else 0,
    }


def epoch_comparison(vals: list[float], steps: list[int]) -> dict:
    """Compare metrics across epoch boundaries."""
    result = {}
    for epoch_frac, label in [(0.5, "e0.5"), (1.0, "e1.0"), (1.5, "e1.5")]:
        target_step = int(epoch_frac * STEPS_PER_EPOCH)
        # Find closest measurement
        dists = [abs(s - target_step) for s in steps]
        if not dists:
            continue
        idx = int(np.argmin(dists))
        if dists[idx] < 2000 and idx < len(vals):  # within 2000 steps
            result[label] = {"step": steps[idx], "value": vals[idx]}
    return result


def fmt(v: float, width: int = 7) -> str:
    if abs(v) < 0.001:
        return f"{v:>{width}.5f}"
    elif abs(v) < 100:
        return f"{v:>{width}.3f}"
    else:
        return f"{v:>{width}.1f}"


def filter_after(mh: dict, steps: list[int], after_step: int) -> tuple[dict, list[int]]:
    """Filter metric history to only include data after a given step."""
    mask = [i for i, s in enumerate(steps) if s >= after_step]
    if not mask:
        return mh, steps
    filtered_steps = [steps[i] for i in mask]
    filtered_mh = {}
    for k, vals in mh.items():
        filtered_mh[k] = [vals[i] for i in mask if i < len(vals)]
    return filtered_mh, filtered_steps


def main():
    import argparse
    parser = argparse.ArgumentParser(description="NCA training dashboard")
    parser.add_argument("checkpoint", nargs="?", default="checkpoints/nca_phase1/nca_resume.pt")
    parser.add_argument("--after", type=int, default=0,
                        help="Only show data after this step (e.g. --after 500 to skip warmup)")
    args = parser.parse_args()

    path = args.checkpoint
    mh, steps, current_step = load(path)

    if args.after > 0:
        mh, steps = filter_after(mh, steps, args.after)

    epoch = current_step / STEPS_PER_EPOCH

    # ── Header ──
    after_label = f"  |  filtered: step >= {args.after}" if args.after > 0 else ""
    print("=" * 100)
    print(f"  NCA DASHBOARD  |  step {current_step}  |  epoch {epoch:.2f}  |  {len(steps)} measurements{after_label}")
    print(f"  checkpoint: {path}")
    print("=" * 100)

    # ── Loss & eval ──
    print("\n── LOSS ─────────────────────────────────────────────────────────")
    for key in ["loss", "eval_loss"]:
        if key not in mh:
            continue
        t = trend(mh[key], steps)
        ep = epoch_comparison(mh[key], steps)
        ep_str = "  ".join(f"{k}={fmt(v['value'])}" for k, v in ep.items())
        print(f"  {key:<15s}: {fmt(t['first'])} → {fmt(t['last'])}  ({t['direction']} {t['strength']}, {t['delta_pct']:+.2f}%)  |  {ep_str}")

    # ── Gradient health ──
    grad_keys = ["grad_norm", "model_grad_norm", "head_grad_norm"]
    available_grad = [k for k in grad_keys if k in mh and mh[k]]
    if available_grad:
        print("\n── GRADIENTS ────────────────────────────────────────────────────")
        for key in available_grad:
            t = trend(mh[key], steps[:len(mh[key])])
            print(f"  {key:<20s}: {fmt(t['last'])}  (mean={fmt(t['mean'])}, std={fmt(t['std'])}, {t['direction']} {t['strength']})")

    # ── RankMe ──
    if "geo/rankme_last" in mh:
        print("\n── RANKME (effective dimensionality) ────────────────────────────")
        t = trend(mh["geo/rankme_last"], steps)
        ep = epoch_comparison(mh["geo/rankme_last"], steps)
        ep_str = "  ".join(f"{k}={fmt(v['value'])}" for k, v in ep.items())
        print(f"  rankme: {fmt(t['first'])} → {fmt(t['last'])}  ({t['direction']} {t['strength']}, {t['delta_pct']:+.1f}%)  |  {ep_str}")
        # Last 5 values
        last5 = mh["geo/rankme_last"][-5:]
        last5_steps = steps[-5:]
        print(f"  recent: " + "  ".join(f"s{s}={v:.1f}" for s, v in zip(last5_steps, last5)))

    # ── Anisotropy per layer ──
    layers = []
    for i in range(20):
        k = f"geo/layer_{i}/anisotropy"
        if k in mh:
            layers.append(i)

    if layers:
        print("\n── ANISOTROPY (per-layer representation specialization) ─────────")
        print(f"  {'layer':<8s} | {'first':>7s} | {'now':>7s} | {'min':>7s} | {'max':>7s} | {'trend':>12s} | {'delta%':>7s} | epoch snapshots")
        print("  " + "-" * 95)
        for li in layers:
            k = f"geo/layer_{li}/anisotropy"
            t = trend(mh[k], steps)
            ep = epoch_comparison(mh[k], steps)
            ep_str = "  ".join(f"{label}={fmt(v['value'])}" for label, v in ep.items())
            print(f"  L{li:<6d} | {fmt(t['first'])} | {fmt(t['last'])} | {fmt(t['min'])} | {fmt(t['max'])} | {t['direction']:<4s} {t['strength']:<6s} | {t['delta_pct']:>+6.1f}% | {ep_str}")

    # ── Attention entropy per layer ──
    attn_layers = [i for i in range(20) if f"geo/layer_{i}/attn_entropy_mean" in mh]
    if attn_layers:
        print("\n── ATTENTION ENTROPY (lower = more specialized heads) ──────────")
        print(f"  {'layer':<8s} | {'first':>7s} | {'now':>7s} | {'trend':>12s} | {'delta%':>7s} | epoch snapshots")
        print("  " + "-" * 85)
        for li in attn_layers:
            k = f"geo/layer_{li}/attn_entropy_mean"
            t = trend(mh[k], steps)
            ep = epoch_comparison(mh[k], steps)
            ep_str = "  ".join(f"{label}={fmt(v['value'])}" for label, v in ep.items())
            print(f"  L{li:<6d} | {fmt(t['first'])} | {fmt(t['last'])} | {t['direction']:<4s} {t['strength']:<6s} | {t['delta_pct']:>+6.1f}% | {ep_str}")

    # ── Stable rank per layer (q_proj as representative) ──
    sr_layers = [i for i in range(20) if f"geo/layer_{i}/stable_rank_q_proj" in mh]
    if sr_layers:
        print("\n── STABLE RANK q_proj (weight-space expressiveness) ─────────────")
        print(f"  {'layer':<8s} | {'first':>7s} | {'now':>7s} | {'trend':>12s} | {'delta%':>7s}")
        print("  " + "-" * 60)
        for li in sr_layers:
            k = f"geo/layer_{li}/stable_rank_q_proj"
            t = trend(mh[k], steps)
            print(f"  L{li:<6d} | {fmt(t['first'])} | {fmt(t['last'])} | {t['direction']:<4s} {t['strength']:<6s} | {t['delta_pct']:>+6.1f}%")
        # Also show gate_proj for L11 (divergent in earlier analysis)
        k11g = "geo/layer_11/stable_rank_gate_proj"
        if k11g in mh:
            t = trend(mh[k11g], steps)
            print(f"  L11 gate | {fmt(t['first'])} | {fmt(t['last'])} | {t['direction']:<4s} {t['strength']:<6s} | {t['delta_pct']:>+6.1f}%")

    # ── Reversal count by epoch half ──
    print("\n── REVERSAL DENSITY (direction flips per epoch-half) ────────────")
    important_keys = [k for k in mh if "anisotropy" in k or "attn_entropy_mean" in k or k == "geo/rankme_last"]
    epoch_halves = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]
    for start_ep, end_ep in epoch_halves:
        s0 = int(start_ep * STEPS_PER_EPOCH)
        s1 = int(end_ep * STEPS_PER_EPOCH)
        count = 0
        for k in important_keys:
            vals = mh[k]
            indices = [i for i, s in enumerate(steps) if s0 <= s <= s1 and i < len(vals)]
            if len(indices) < 3:
                continue
            sub = [vals[i] for i in indices]
            deltas = [sub[j+1] - sub[j] for j in range(len(sub)-1)]
            for j in range(1, len(deltas)):
                if deltas[j-1] * deltas[j] < 0:
                    count += 1
        data_in_range = any(s0 <= s <= s1 for s in steps)
        if data_in_range:
            print(f"  epoch {start_ep:.1f}-{end_ep:.1f}: {count} reversals across {len(important_keys)} key metrics")

    # ── Summary assessment ──
    print("\n── ASSESSMENT ───────────────────────────────────────────────────")
    eval_t = trend(mh["eval_loss"], steps) if "eval_loss" in mh else None
    if eval_t and eval_t["strength"] == "strong" and eval_t["direction"] == "DOWN":
        print("  eval_loss: still improving (no overfitting)")
    elif eval_t:
        print(f"  eval_loss: {eval_t['direction']} {eval_t['strength']} — watch closely")

    # Check for late-layer collapse signal
    l11_anis = mh.get("geo/layer_11/anisotropy")
    if l11_anis:
        t11 = trend(l11_anis, steps)
        if t11["direction"] == "DOWN" and t11["strength"] == "strong":
            print(f"  L11 anisotropy: declining {t11['delta_pct']:+.1f}% — late-layer de-specialization (AttnRes signal)")

    rm = mh.get("geo/rankme_last")
    if rm:
        rm_t = trend(rm, steps)
        if rm_t["direction"] == "UP" and rm_t["strength"] == "strong":
            print(f"  RankMe: expanding {rm_t['delta_pct']:+.1f}% — Muon preserving dimensional diversity")

    print()


if __name__ == "__main__":
    main()
