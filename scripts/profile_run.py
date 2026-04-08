#!/usr/bin/env python3
"""Metis deep profiling harness.

Runs a full but short training iteration with comprehensive profiling
to identify bottlenecks. All computational parameters match production
exactly — only game count and training steps are reduced.

Profiles both GPU (inference + training) and CPU (pipeline overhead),
producing Chrome traces, batch statistics, and a summary report.

Produces:
    profiles/<timestamp>/
        selfplay_trace.json    Chrome trace for self-play GPU activity
        training_trace.json    Chrome trace for training step GPU activity
        report.txt             Human-readable summary with bottleneck analysis
        metrics.json           Machine-readable metrics for comparison
        gpu_utilization.csv    GPU SM utilization over time (if pynvml available)

Usage:
    # Profile on GPU 7 while training runs on GPU 0
    CUDA_VISIBLE_DEVICES=7 python scripts/profile_run.py \\
        --config configs/selfplay_profile.yaml

    # Custom output directory
    CUDA_VISIBLE_DEVICES=7 python scripts/profile_run.py \\
        --config configs/selfplay_profile.yaml \\
        --output profiles/experiment_1

    # Load a specific checkpoint for realistic model behavior
    CUDA_VISIBLE_DEVICES=7 python scripts/profile_run.py \\
        --config configs/selfplay_profile.yaml \\
        --checkpoint checkpoints/selfplay/iter_000050.pt

    # For CPU/MCTS flamegraph, run py-spy in another terminal:
    py-spy record --pid <PID> --subprocesses -o mcts_flame.svg
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.transformer import ChessModelConfig, ChessTransformer, MODEL_CONFIGS
from src.chess.self_play import (
    SelfPlayConfig,
    ParallelSelfPlay,
    _DoubleBufferedEvaluator,
)
from src.training.replay_buffer import ReplayBuffer
from src.training.train import compute_loss
from src.training.muon import build_hybrid_optimizer

logger = logging.getLogger(__name__)


# ── NVML GPU utilization sampler (optional) ───────────────────────────────


class GPUUtilizationSampler:
    """Background thread that samples GPU SM + memory utilization via NVML.

    Falls back gracefully if pynvml is not installed.
    """

    def __init__(self, device_index: int, interval_s: float = 0.1):
        self._device_index = device_index
        self._interval = interval_s
        self._samples: list[tuple[float, int, int]] = []  # (time, gpu%, mem%)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._handle = None
        self._available = False

        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._pynvml = pynvml
            self._available = True
        except Exception:
            pass

    @property
    def available(self) -> bool:
        return self._available

    def start(self) -> None:
        if not self._available:
            return
        self._stop.clear()
        self._samples.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[tuple[float, int, int]]:
        if not self._available or self._thread is None:
            return []
        self._stop.set()
        self._thread.join(timeout=2.0)
        return list(self._samples)

    def _sample_loop(self) -> None:
        t0 = time.monotonic()
        while not self._stop.is_set():
            try:
                util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                self._samples.append(
                    (time.monotonic() - t0, util.gpu, util.memory)
                )
            except Exception:
                pass
            self._stop.wait(timeout=self._interval)


# ── Evaluator instrumentation ─────────────────────────────────────────────


def _install_evaluator_hooks(device: torch.device) -> dict:
    """Monkey-patch _DoubleBufferedEvaluator to collect batch statistics.

    Uses CUDA events for accurate GPU timing without pipeline distortion.
    Events are recorded asynchronously; timing is read after self-play ends.

    Returns a dict that accumulates profiling data (mutated in-place by hook).
    """
    stats: dict = {
        "batch_sizes": [],
        "cuda_events": [],  # list of (start_event, end_event)
        "wall_times": [],  # wall-clock per-batch (includes CPU overhead)
    }
    original_fn = _DoubleBufferedEvaluator._evaluate_batch

    def _hooked_evaluate_batch(
        self: _DoubleBufferedEvaluator,
        board_tensor: torch.Tensor,
        max_sub_batch: int = 4096,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = board_tensor.shape[0]
        stats["batch_sizes"].append(n)

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        start_evt.record(torch.cuda.current_stream(device))
        t0 = time.monotonic()

        result = original_fn(self, board_tensor, max_sub_batch)

        end_evt.record(torch.cuda.current_stream(device))
        stats["cuda_events"].append((start_evt, end_evt))
        stats["wall_times"].append(time.monotonic() - t0)

        return result

    _DoubleBufferedEvaluator._evaluate_batch = _hooked_evaluate_batch

    return stats


def _finalize_evaluator_stats(stats: dict) -> dict:
    """Synchronize CUDA events and compute GPU timing from evaluator hooks."""
    torch.cuda.synchronize()

    gpu_times_ms: list[float] = []
    for start_evt, end_evt in stats["cuda_events"]:
        try:
            gpu_times_ms.append(start_evt.elapsed_time(end_evt))
        except RuntimeError:
            # Event may be invalid if evaluator was interrupted
            pass

    bs = np.array(stats["batch_sizes"]) if stats["batch_sizes"] else np.array([0])
    gpu_ms = np.array(gpu_times_ms) if gpu_times_ms else np.array([0.0])
    wall = np.array(stats["wall_times"]) if stats["wall_times"] else np.array([0.0])

    return {
        "batch_sizes": bs,
        "gpu_times_ms": gpu_ms,
        "wall_times_s": wall,
        "total_gpu_s": gpu_ms.sum() / 1000.0,
        "total_wall_s": wall.sum(),
        "total_evals": int(bs.sum()),
        "num_batches": len(bs),
        "mean_batch_size": float(bs.mean()) if len(bs) > 0 else 0.0,
        "median_batch_size": float(np.median(bs)) if len(bs) > 0 else 0.0,
        "max_batch_size": int(bs.max()) if len(bs) > 0 else 0,
        "p10_batch_size": float(np.percentile(bs, 10)) if len(bs) > 1 else 0.0,
        "p90_batch_size": float(np.percentile(bs, 90)) if len(bs) > 1 else 0.0,
        "mean_gpu_ms": float(gpu_ms.mean()) if len(gpu_ms) > 0 else 0.0,
        "mean_wall_ms": float(wall.mean() * 1000) if len(wall) > 0 else 0.0,
    }


# ── Config loading ────────────────────────────────────────────────────────


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Model setup ───────────────────────────────────────────────────────────


def setup_model(
    config: dict, device: torch.device
) -> tuple[ChessTransformer, ChessModelConfig]:
    """Create model from config, matching train.py setup exactly."""
    model_cfg = config.get("model", {})
    model_kwargs = MODEL_CONFIGS.get(model_cfg.get("size", "medium"), {})
    mc = ChessModelConfig(**model_kwargs)

    if model_cfg.get("attn_res_boundaries"):
        mc.attn_res = True
        mc.attn_res_boundaries = [
            int(x) for x in model_cfg["attn_res_boundaries"].split(",")
        ]
    if model_cfg.get("activation_checkpointing"):
        mc.activation_checkpointing = True

    model = ChessTransformer(mc)
    model = model.to(device)
    return model, mc


def load_checkpoint(
    model: ChessTransformer, config: dict, explicit_path: Optional[str], device: torch.device
) -> Optional[int]:
    """Load checkpoint if available. Returns iteration number or None."""
    ckpt_path: Optional[Path] = None

    if explicit_path:
        ckpt_path = Path(explicit_path)
    else:
        ckpt_dir = Path(
            config.get("checkpoint", {}).get("checkpoint_dir", "checkpoints/selfplay")
        )
        latest = ckpt_dir / "latest.pt"
        if latest.exists():
            ckpt_path = latest

    if ckpt_path is None or not ckpt_path.exists():
        return None

    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Filter shape mismatches for cross-version compatibility
    ckpt_state = ckpt["model"]
    model_state = model.state_dict()
    filtered = {
        k: v
        for k, v in ckpt_state.items()
        if k in model_state and v.shape == model_state[k].shape
    }
    skipped = set(ckpt_state.keys()) - set(filtered.keys())
    if skipped:
        print(f"  Skipped {len(skipped)} keys with shape mismatch")
    model.load_state_dict(filtered, strict=False)

    iteration = ckpt.get("iteration", None)
    print(f"  Loaded (iteration {iteration})")
    del ckpt
    return iteration


# ── Training step profiling ───────────────────────────────────────────────


def profile_training_steps(
    model: torch.nn.Module,
    replay_buffer: ReplayBuffer,
    device: torch.device,
    batch_size: int,
    num_steps: int,
    output_dir: Path,
    model_config: ChessModelConfig,
) -> dict:
    """Profile training steps with per-phase timing and torch.profiler trace."""
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    muon_opt, adamw_opt = build_hybrid_optimizer(
        raw_model,
        muon_lr=0.02,
        muon_momentum=0.95,
        muon_weight_decay=0.01,
        muon_ns_iterations=5,
        muon_ns_coefficients="gram_ns",
        adamw_lr=3e-4,
        adamw_betas=(0.9, 0.95),
        adamw_weight_decay=0.1,
    )

    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    data_times: list[float] = []
    fwd_times: list[float] = []
    bwd_times: list[float] = []
    opt_times: list[float] = []

    # torch.profiler: skip first 5 steps (warmup), profile next 20
    active_steps = min(20, max(1, num_steps - 5))

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        schedule=torch.profiler.schedule(
            wait=2,
            warmup=3,
            active=active_steps,
            repeat=1,
        ),
    ) as prof:
        train_start = time.time()

        for step in range(num_steps):
            # ── Data loading ──
            torch.cuda.synchronize(device)
            t0 = time.monotonic()

            boards, policies, values, materials, activities = replay_buffer.sample(
                batch_size
            )
            boards = boards.to(device)
            policies = policies.to(device)
            values = values.to(device)
            materials = materials.to(device)
            activities = activities.to(device)

            torch.cuda.synchronize(device)
            t1 = time.monotonic()
            data_times.append(t1 - t0)

            # ── Forward ──
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                policy_logits, wdl_logits, mat_pred, act_pred = model(boards)
                losses = compute_loss(
                    policy_logits,
                    wdl_logits,
                    mat_pred,
                    act_pred,
                    policies,
                    values,
                    materials,
                    activities,
                    z_loss_weight=model_config.z_loss_weight,
                )

            torch.cuda.synchronize(device)
            t2 = time.monotonic()
            fwd_times.append(t2 - t1)

            # ── Backward ──
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            torch.cuda.synchronize(device)
            t3 = time.monotonic()
            bwd_times.append(t3 - t2)

            # ── Optimizer ──
            muon_opt.step()
            adamw_opt.step()
            muon_opt.zero_grad(set_to_none=True)
            adamw_opt.zero_grad(set_to_none=True)

            torch.cuda.synchronize(device)
            t4 = time.monotonic()
            opt_times.append(t4 - t3)

            prof.step()

        train_time = time.time() - train_start

    # Save trace
    trace_path = output_dir / "training_trace.json"
    prof.export_chrome_trace(str(trace_path))

    vram_peak = torch.cuda.max_memory_allocated(device) / 1e9

    # Exclude first 5 steps (warmup) for averages
    skip = min(5, len(data_times) - 1)
    return {
        "total_time_s": train_time,
        "num_steps": num_steps,
        "steps_per_sec": num_steps / max(train_time, 1e-9),
        "vram_peak_gb": vram_peak,
        "trace_path": str(trace_path),
        "per_step": {
            "data_ms": float(np.mean(data_times[skip:]) * 1000),
            "forward_ms": float(np.mean(fwd_times[skip:]) * 1000),
            "backward_ms": float(np.mean(bwd_times[skip:]) * 1000),
            "optimizer_ms": float(np.mean(opt_times[skip:]) * 1000),
            "total_ms": float(
                (
                    np.mean(data_times[skip:])
                    + np.mean(fwd_times[skip:])
                    + np.mean(bwd_times[skip:])
                    + np.mean(opt_times[skip:])
                )
                * 1000
            ),
        },
        "all_step_times_ms": [
            (d + f + b + o) * 1000
            for d, f, b, o in zip(data_times, fwd_times, bwd_times, opt_times)
        ],
    }


# ── Report generation ─────────────────────────────────────────────────────


def generate_report(
    *,
    gpu_name: str,
    gpu_mem_gb: float,
    param_count: int,
    config: dict,
    sp_config: SelfPlayConfig,
    games_per_iter: int,
    batch_size: int,
    train_steps: int,
    warmup_time: float,
    warmup_games: int,
    sp_time: float,
    sp_games: int,
    sp_positions: int,
    sp_vram_gb: float,
    eval_stats: dict,
    gpu_samples: list[tuple[float, int, int]],
    train_results: Optional[dict],
    output_dir: Path,
) -> str:
    """Build the profiling report string."""
    lines: list[str] = []
    w = lines.append

    def section(title: str) -> None:
        w(f"\n{'─' * 60}")
        w(f"  {title}")
        w(f"{'─' * 60}")

    w("=" * 60)
    w("  METIS PROFILING REPORT")
    w(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    w("=" * 60)

    w(f"\nHardware:  {gpu_name} ({gpu_mem_gb:.0f} GB VRAM)")
    w(f"Model:     medium ({param_count:,} params)")
    w(f"Config:    {config.get('_config_path', 'selfplay_profile.yaml')}")

    # ── Iteration time estimate ──
    section("ITERATION TIME BREAKDOWN (estimated for v9: 256 games, 1000 steps)")

    game_ratio = 256.0 / max(games_per_iter, 1)
    step_ratio = 1000.0 / max(train_steps, 1)
    est_sp = sp_time * game_ratio
    est_tr = train_results["total_time_s"] * step_ratio if train_results else 0
    est_total = est_sp + est_tr

    w(f"  Self-play (256 games):  ~{est_sp:6.0f}s  ({est_sp / max(est_total, 1e-9) * 100:5.1f}%)")
    if train_results:
        w(f"  Training (1000 steps):  ~{est_tr:6.0f}s  ({est_tr / max(est_total, 1e-9) * 100:5.1f}%)")
    w(f"  Estimated iter total:   ~{est_total:6.0f}s")

    # ── Self-play breakdown ──
    section(f"SELF-PLAY ({sp_games} games, {sp_time:.1f}s)")

    total_gpu_s = eval_stats["total_gpu_s"]
    gpu_idle_s = sp_time - total_gpu_s

    w(f"  GPU inference time:    {total_gpu_s:7.2f}s  ({total_gpu_s / max(sp_time, 1e-9) * 100:5.1f}%)")
    w(f"  GPU idle / CPU work:   {gpu_idle_s:7.2f}s  ({gpu_idle_s / max(sp_time, 1e-9) * 100:5.1f}%)")
    w(f"  Peak VRAM:             {sp_vram_gb:.2f} GB  ({sp_vram_gb / gpu_mem_gb * 100:.1f}% of {gpu_mem_gb:.0f} GB)")

    w(f"\n  Batching statistics:")
    w(f"    Total batches:       {eval_stats['num_batches']}")
    w(f"    Total NN evals:      {eval_stats['total_evals']}")
    w(f"    Mean batch size:     {eval_stats['mean_batch_size']:.1f} positions")
    w(f"    Median batch size:   {eval_stats['median_batch_size']:.0f} positions")
    w(f"    Max batch size:      {eval_stats['max_batch_size']} positions")
    w(f"    P10 / P90 batch:     {eval_stats['p10_batch_size']:.0f} / {eval_stats['p90_batch_size']:.0f}")
    w(f"    Mean GPU time/batch: {eval_stats['mean_gpu_ms']:.2f} ms")
    evals_per_sec = eval_stats["total_evals"] / max(sp_time, 1e-9)
    w(f"    Evals/sec:           {evals_per_sec:,.0f}")

    # ── GPU utilization (NVML) ──
    if gpu_samples:
        gpu_utils = [s[1] for s in gpu_samples]
        mem_utils = [s[2] for s in gpu_samples]
        section("GPU UTILIZATION (NVML, 10 Hz sampling during self-play)")
        w(f"  SM utilization:   mean {np.mean(gpu_utils):5.1f}%  "
          f"median {np.median(gpu_utils):5.1f}%  "
          f"P90 {np.percentile(gpu_utils, 90):5.1f}%  "
          f"max {np.max(gpu_utils):5.1f}%")
        w(f"  Memory bandwidth: mean {np.mean(mem_utils):5.1f}%  "
          f"median {np.median(mem_utils):5.1f}%  "
          f"max {np.max(mem_utils):5.1f}%")
        # Fraction of time GPU was truly idle (0%)
        idle_frac = sum(1 for u in gpu_utils if u == 0) / max(len(gpu_utils), 1)
        w(f"  Fully idle:       {idle_frac * 100:.1f}% of samples")

    # ── Training breakdown ──
    if train_results:
        tr = train_results
        ps = tr["per_step"]
        total_step_ms = ps["total_ms"]

        section(f"TRAINING ({tr['num_steps']} steps, batch {batch_size}, {tr['total_time_s']:.1f}s)")
        w(f"  Per-step breakdown (avg, excluding first 5 warmup steps):")
        w(f"    Data loading:    {ps['data_ms']:7.2f} ms  ({ps['data_ms'] / total_step_ms * 100:5.1f}%)")
        w(f"    Forward pass:    {ps['forward_ms']:7.2f} ms  ({ps['forward_ms'] / total_step_ms * 100:5.1f}%)")
        w(f"    Backward pass:   {ps['backward_ms']:7.2f} ms  ({ps['backward_ms'] / total_step_ms * 100:5.1f}%)")
        w(f"    Optimizer step:  {ps['optimizer_ms']:7.2f} ms  ({ps['optimizer_ms'] / total_step_ms * 100:5.1f}%)")
        w(f"    Total:           {total_step_ms:7.2f} ms  ({1000 / total_step_ms:.1f} steps/s)")
        w(f"  Peak VRAM:         {tr['vram_peak_gb']:.2f} GB  ({tr['vram_peak_gb'] / gpu_mem_gb * 100:.1f}% of {gpu_mem_gb:.0f} GB)")

    # ── Bottleneck analysis ──
    section("BOTTLENECK ANALYSIS")

    gpu_util_pct = total_gpu_s / max(sp_time, 1e-9) * 100

    if gpu_util_pct < 30:
        w(f"  [!!] GPU utilization during self-play is VERY LOW ({gpu_util_pct:.0f}%)")
        w(f"       GPU sits idle {100 - gpu_util_pct:.0f}% of self-play time, waiting for workers.")
        w(f"       This is a CPU-bound pipeline: MCTS workers can't feed the GPU fast enough.")
        w(f"       Consider:")
        w(f"         - More parallel games (currently {sp_config.num_parallel})")
        w(f"         - Fewer MCTS sims (currently {sp_config.mcts_simulations}) if signal allows")
        w(f"         - MCTS optimization (Numba hot paths, tree memory layout)")
        w(f"         - Batching more virtual leaves (currently {sp_config.num_virtual_leaves})")
    elif gpu_util_pct < 60:
        w(f"  [~]  GPU utilization during self-play is MODERATE ({gpu_util_pct:.0f}%)")
        w(f"       Some headroom — consider more parallel games or virtual leaves.")
    else:
        w(f"  [ok] GPU utilization during self-play is GOOD ({gpu_util_pct:.0f}%)")

    if eval_stats["mean_batch_size"] < sp_config.num_parallel * 0.3:
        w(f"  [!!] Mean batch size ({eval_stats['mean_batch_size']:.0f}) is far below "
          f"parallel games ({sp_config.num_parallel})")
        w(f"       Workers aren't generating eval requests fast enough to fill batches.")

    if train_results:
        tr_vram_pct = train_results["vram_peak_gb"] / gpu_mem_gb * 100
        if tr_vram_pct < 15:
            w(f"  [!!] Training VRAM ({train_results['vram_peak_gb']:.1f} GB) is only "
              f"{tr_vram_pct:.0f}% of available ({gpu_mem_gb:.0f} GB)")
            w(f"       Batch size could be increased significantly from {batch_size}.")
        elif tr_vram_pct < 40:
            w(f"  [~]  Training VRAM usage is moderate ({tr_vram_pct:.0f}%)")
            w(f"       Room to increase batch size if gradient quality benefits.")

    sp_vram_pct = sp_vram_gb / gpu_mem_gb * 100
    if sp_vram_pct < 10:
        w(f"  [!!] Inference VRAM ({sp_vram_gb:.1f} GB) is only {sp_vram_pct:.0f}% of "
          f"available ({gpu_mem_gb:.0f} GB)")
        w(f"       The model is small relative to the hardware.")
        w(f"       Consider: larger model, bigger inference batches, or model ensembling.")

    total_vram = max(sp_vram_gb, train_results["vram_peak_gb"] if train_results else 0)
    w(f"\n  Overall peak VRAM: {total_vram:.1f} GB / {gpu_mem_gb:.0f} GB "
      f"({total_vram / gpu_mem_gb * 100:.0f}% utilized)")

    # ── CPU profiling hint ──
    section("CPU / MCTS PROFILING (run separately)")
    w(f"  The torch.profiler traces cover GPU activity. For CPU/MCTS bottleneck")
    w(f"  breakdown (find_leaf, expand, backup, board encoding, queue waits),")
    w(f"  use py-spy in a separate terminal during the self-play phase:")
    w(f"")
    w(f"    # Flamegraph of all processes (main + workers)")
    w(f"    py-spy record --pid $(pgrep -f profile_run) --subprocesses \\")
    w(f"        -o {output_dir}/mcts_flame.svg")
    w(f"")
    w(f"    # Or attach to a single worker for focused view")
    w(f"    py-spy top --pid <WORKER_PID>")

    # ── Output files ──
    section("OUTPUT FILES")
    w(f"  {output_dir}/selfplay_trace.json   -> chrome://tracing or ui.perfetto.dev")
    if train_results:
        w(f"  {output_dir}/training_trace.json   -> chrome://tracing or ui.perfetto.dev")
    w(f"  {output_dir}/report.txt")
    w(f"  {output_dir}/metrics.json")
    if gpu_samples:
        w(f"  {output_dir}/gpu_utilization.csv")

    w(f"\n{'=' * 60}")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Metis deep profiler — isomorphic benchmark for bottleneck analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              CUDA_VISIBLE_DEVICES=7 python scripts/profile_run.py --config configs/selfplay_profile.yaml
              python scripts/profile_run.py --config configs/selfplay_profile.yaml --checkpoint checkpoints/selfplay/iter_000050.pt
        """),
    )
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML config (e.g. configs/selfplay_profile.yaml)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: profiles/<timestamp>)",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Explicit checkpoint path (default: latest from config's checkpoint_dir)",
    )
    p.add_argument(
        "--warmup_games",
        type=int,
        default=32,
        help="Games for warmup iteration (JIT/torch.compile, default: 32)",
    )
    p.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training profiling (self-play only)",
    )
    p.add_argument(
        "--skip_selfplay",
        action="store_true",
        help="Skip self-play profiling (training only, needs existing buffer data)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    config["_config_path"] = args.config

    # Output directory
    if args.output is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"profiles/{timestamp}")
    else:
        output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Device setup ──
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Profiling requires a GPU.")
        sys.exit(1)

    device = torch.device("cuda")
    gpu_idx = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem_gb = torch.cuda.get_device_properties(device).total_memory / 1e9

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"{'=' * 60}")
    print(f"  Metis Deep Profiler")
    print(f"  GPU: {gpu_name} ({gpu_mem_gb:.0f} GB)")
    print(f"  Config: {args.config}")
    print(f"  Output: {output_dir}")
    print(f"  PID: {os.getpid()} (for py-spy attachment)")
    print(f"{'=' * 60}")

    # ── Model setup ──
    print(f"\nSetting up model...")
    model, model_config = setup_model(config, device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  {param_count:,} parameters")

    # Load checkpoint
    load_checkpoint(model, config, args.checkpoint, device)

    # torch.compile
    model_cfg = config.get("model", {})
    if model_cfg.get("compile", False):
        print("  Compiling model with torch.compile...")
        model = torch.compile(model)

    # ── Self-play config ──
    sp_cfg = config.get("self_play", {})
    sp_config = SelfPlayConfig(
        num_parallel=sp_cfg.get("num_parallel_games", 192),
        num_workers=sp_cfg.get("num_workers", 96),
        mcts_simulations=sp_cfg.get("mcts_simulations", 800),
        temperature_threshold=sp_cfg.get("temperature_threshold", 30),
        dirichlet_epsilon=sp_cfg.get("dirichlet_epsilon", 0.25),
        num_virtual_leaves=sp_cfg.get("num_virtual_leaves", 8),
    )

    games_per_iter = sp_cfg.get("games_per_iter", 32)

    tr_cfg = config.get("training", {})
    batch_size = tr_cfg.get("batch_size", 1024)
    train_steps = tr_cfg.get("train_steps_per_iter", 50)

    sp_engine = ParallelSelfPlay(model, sp_config, device)
    replay_buffer = ReplayBuffer(capacity=tr_cfg.get("buffer_size", 1_000_000))

    # NVML sampler
    gpu_sampler = GPUUtilizationSampler(gpu_idx)
    if gpu_sampler.available:
        print(f"  NVML GPU utilization sampling: enabled (10 Hz)")
    else:
        print(f"  NVML GPU utilization sampling: unavailable (install pynvml for this)")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: Warmup
    # ══════════════════════════════════════════════════════════════════
    if not args.skip_selfplay:
        print(f"\n{'=' * 60}")
        print(f"  PHASE 1: Warmup ({args.warmup_games} games)")
        print(f"  torch.compile + Numba JIT compilation happens here")
        print(f"{'=' * 60}")

        model.eval()
        warmup_start = time.time()
        warmup_games_list = sp_engine.generate_games(args.warmup_games)
        warmup_time = time.time() - warmup_start

        for g in warmup_games_list:
            replay_buffer.add_game(g)

        warmup_positions = sum(len(g) for g in warmup_games_list)
        print(f"  {len(warmup_games_list)} games, {warmup_positions} positions in {warmup_time:.1f}s")
        print(f"  Buffer: {len(replay_buffer)} positions")
    else:
        warmup_time = 0.0
        warmup_games_list = []

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: Profiled Self-Play
    # ══════════════════════════════════════════════════════════════════
    sp_time = 0.0
    sp_vram_gb = 0.0
    games = []
    eval_stats_final: dict = {}
    gpu_samples: list = []

    if not args.skip_selfplay:
        print(f"\n{'=' * 60}")
        print(f"  PHASE 2: Profiled Self-Play ({games_per_iter} games)")
        print(f"  Workers: {sp_config.num_workers}  |  Parallel: {sp_config.num_parallel}")
        print(f"  MCTS sims: {sp_config.mcts_simulations}  |  Virtual leaves: {sp_config.num_virtual_leaves}")
        print(f"{'=' * 60}")

        # Install evaluator hooks for batch statistics
        eval_hook_stats = _install_evaluator_hooks(device)

        model.eval()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        # Start NVML sampling
        gpu_sampler.start()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
        ) as sp_prof:
            sp_start = time.time()
            games = sp_engine.generate_games(games_per_iter)
            sp_time = time.time() - sp_start

        # Stop sampling, finalize
        gpu_samples = gpu_sampler.stop()
        eval_stats_final = _finalize_evaluator_stats(eval_hook_stats)
        sp_vram_gb = torch.cuda.max_memory_allocated(device) / 1e9

        # Save trace
        sp_trace_path = output_dir / "selfplay_trace.json"
        sp_prof.export_chrome_trace(str(sp_trace_path))

        for g in games:
            replay_buffer.add_game(g)

        sp_positions = sum(len(g) for g in games)
        print(f"  {len(games)} games, {sp_positions} positions in {sp_time:.1f}s")
        print(f"  {eval_stats_final['total_evals']} NN evals in {eval_stats_final['num_batches']} batches")
        print(f"  Peak VRAM: {sp_vram_gb:.2f} GB")
        print(f"  Trace: {sp_trace_path}")
    else:
        sp_positions = 0

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: Profiled Training
    # ══════════════════════════════════════════════════════════════════
    train_results: Optional[dict] = None

    if not args.skip_training:
        if len(replay_buffer) < batch_size:
            print(f"\n  WARNING: Buffer ({len(replay_buffer)}) < batch_size ({batch_size})")
            print(f"  Reducing batch_size to {len(replay_buffer)} for profiling")
            batch_size = max(1, len(replay_buffer))

        if len(replay_buffer) > 0:
            print(f"\n{'=' * 60}")
            print(f"  PHASE 3: Profiled Training ({train_steps} steps, batch {batch_size})")
            print(f"  Buffer: {len(replay_buffer)} positions")
            print(f"{'=' * 60}")

            train_results = profile_training_steps(
                model=model,
                replay_buffer=replay_buffer,
                device=device,
                batch_size=batch_size,
                num_steps=train_steps,
                output_dir=output_dir,
                model_config=model_config,
            )

            ps = train_results["per_step"]
            print(f"  {train_steps} steps in {train_results['total_time_s']:.1f}s "
                  f"({train_results['steps_per_sec']:.1f} steps/s)")
            print(f"  Per-step: data {ps['data_ms']:.1f}ms | fwd {ps['forward_ms']:.1f}ms | "
                  f"bwd {ps['backward_ms']:.1f}ms | opt {ps['optimizer_ms']:.1f}ms")
            print(f"  Peak VRAM: {train_results['vram_peak_gb']:.2f} GB")
            print(f"  Trace: {train_results['trace_path']}")
        else:
            print(f"\n  Skipping training: buffer is empty")

    # ══════════════════════════════════════════════════════════════════
    # REPORT
    # ══════════════════════════════════════════════════════════════════
    report = generate_report(
        gpu_name=gpu_name,
        gpu_mem_gb=gpu_mem_gb,
        param_count=param_count,
        config=config,
        sp_config=sp_config,
        games_per_iter=games_per_iter,
        batch_size=batch_size,
        train_steps=train_steps,
        warmup_time=warmup_time,
        warmup_games=len(warmup_games_list) if not args.skip_selfplay else 0,
        sp_time=sp_time,
        sp_games=len(games),
        sp_positions=sp_positions,
        sp_vram_gb=sp_vram_gb,
        eval_stats=eval_stats_final if eval_stats_final else {
            "total_gpu_s": 0, "num_batches": 0, "total_evals": 0,
            "mean_batch_size": 0, "median_batch_size": 0, "max_batch_size": 0,
            "p10_batch_size": 0, "p90_batch_size": 0, "mean_gpu_ms": 0,
            "mean_wall_ms": 0, "total_wall_s": 0,
        },
        gpu_samples=gpu_samples,
        train_results=train_results,
        output_dir=output_dir,
    )

    print(report)

    # Save report to file
    report_path = output_dir / "report.txt"
    report_path.write_text(report)

    # Save machine-readable metrics
    metrics: dict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": args.config,
        "gpu": gpu_name,
        "gpu_mem_gb": gpu_mem_gb,
        "param_count": param_count,
    }

    if not args.skip_selfplay and eval_stats_final:
        metrics["selfplay"] = {
            "games": len(games),
            "positions": sp_positions,
            "total_time_s": sp_time,
            "vram_peak_gb": sp_vram_gb,
            "gpu_inference_time_s": eval_stats_final["total_gpu_s"],
            "gpu_utilization_pct": eval_stats_final["total_gpu_s"] / max(sp_time, 1e-9) * 100,
            "num_batches": eval_stats_final["num_batches"],
            "total_evals": eval_stats_final["total_evals"],
            "evals_per_sec": eval_stats_final["total_evals"] / max(sp_time, 1e-9),
            "mean_batch_size": eval_stats_final["mean_batch_size"],
            "median_batch_size": eval_stats_final["median_batch_size"],
            "max_batch_size": eval_stats_final["max_batch_size"],
            "batch_sizes": eval_stats_final["batch_sizes"].tolist(),
            "gpu_times_ms": eval_stats_final["gpu_times_ms"].tolist(),
        }

    if gpu_samples:
        gpu_utils = [s[1] for s in gpu_samples]
        metrics["nvml"] = {
            "mean_sm_pct": float(np.mean(gpu_utils)),
            "median_sm_pct": float(np.median(gpu_utils)),
            "max_sm_pct": float(np.max(gpu_utils)),
            "idle_frac": sum(1 for u in gpu_utils if u == 0) / max(len(gpu_utils), 1),
            "num_samples": len(gpu_utils),
        }

    if train_results:
        metrics["training"] = {
            "num_steps": train_results["num_steps"],
            "batch_size": batch_size,
            "total_time_s": train_results["total_time_s"],
            "steps_per_sec": train_results["steps_per_sec"],
            "vram_peak_gb": train_results["vram_peak_gb"],
            "per_step_ms": train_results["per_step"],
        }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    # Save GPU utilization CSV
    if gpu_samples:
        csv_path = output_dir / "gpu_utilization.csv"
        with open(csv_path, "w") as f:
            f.write("time_s,gpu_util_pct,mem_util_pct\n")
            for t, gpu, mem in gpu_samples:
                f.write(f"{t:.3f},{gpu},{mem}\n")

    print(f"\nDone. All outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()
