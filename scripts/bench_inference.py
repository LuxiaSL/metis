#!/usr/bin/env python3
"""Standalone GPU inference benchmark for the chess transformer.

Isolates model forward pass from the MCTS pipeline to measure
raw inference throughput and identify GPU-side optimization targets.

Tests:
  - Batch size scaling (1 to 4096)
  - torch.compile vs eager mode
  - CUDA graphs (fixed batch sizes)
  - Component breakdown (transfer, forward, softmax, copy-back)
  - Warm cache vs cold cache

Usage:
    CUDA_VISIBLE_DEVICES=7 python scripts/bench_inference.py
    CUDA_VISIBLE_DEVICES=7 python scripts/bench_inference.py --checkpoint checkpoints/selfplay/nca_checkpoint.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.transformer import ChessModelConfig, ChessTransformer, MODEL_CONFIGS
from src.chess.board import SEQ_LEN, POLICY_SIZE


def setup_model(
    checkpoint: str | None, device: torch.device, compile: bool
) -> tuple[torch.nn.Module, ChessModelConfig]:
    model_kwargs = MODEL_CONFIGS["medium"]
    config = ChessModelConfig(**model_kwargs)
    config.attn_res = True
    config.attn_res_boundaries = [0, 1, 3, 9]

    model = ChessTransformer(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: medium ({param_count:,} params)")

    if checkpoint and Path(checkpoint).exists():
        print(f"Loading checkpoint: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = ckpt["model"]
        model_state = model.state_dict()
        filtered = {
            k: v for k, v in state.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        model.load_state_dict(filtered, strict=False)
        del ckpt

    model.eval()

    if compile:
        print("Compiling model...")
        model = torch.compile(model)

    return model, config


def make_batch(batch_size: int, device: torch.device) -> torch.Tensor:
    """Create a random input batch matching real board encoding shape.

    Token layout (67 total):
      [0]   castling rights  → vocab 16 (0-15)
      [1]   en passant file  → vocab 9  (0-8)
      [2]   side to move     → vocab 2  (0-1)
      [3-66] piece per square → vocab 13 (0-12)
    """
    batch = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long, device=device)
    batch[:, 0] = torch.randint(0, 16, (batch_size,), device=device)   # castling
    batch[:, 1] = torch.randint(0, 9, (batch_size,), device=device)    # en passant
    batch[:, 2] = torch.randint(0, 2, (batch_size,), device=device)    # side to move
    batch[:, 3:] = torch.randint(0, 13, (batch_size, 64), device=device)  # pieces
    return batch


@torch.no_grad()
def bench_forward(
    model: torch.nn.Module,
    batch: torch.Tensor,
    num_warmup: int = 10,
    num_iters: int = 50,
) -> dict[str, float]:
    """Benchmark just the forward pass (data already on GPU)."""
    # Warmup
    for _ in range(num_warmup):
        model(batch)
    torch.cuda.synchronize()

    # Timed
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iters):
        model(batch)
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    per_iter_ms = total_ms / num_iters
    batch_size = batch.shape[0]

    return {
        "batch_size": batch_size,
        "forward_ms": per_iter_ms,
        "per_position_us": per_iter_ms * 1000 / batch_size,
        "positions_per_sec": batch_size / (per_iter_ms / 1000),
    }


@torch.no_grad()
def bench_full_pipeline(
    model: torch.nn.Module,
    batch_cpu: torch.Tensor,
    device: torch.device,
    num_warmup: int = 10,
    num_iters: int = 50,
) -> dict[str, float]:
    """Benchmark the full evaluator pipeline: transfer → forward → softmax → copy back."""
    batch_size = batch_cpu.shape[0]

    # Warmup
    for _ in range(num_warmup):
        b = batch_cpu.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pl, wdl, _, _ = model(b)
        policies = torch.softmax(pl.float(), dim=-1).cpu().numpy()
        wdl_probs = torch.softmax(wdl.float(), dim=-1)
        values = (wdl_probs[:, 2] - wdl_probs[:, 0]).cpu().numpy()
    torch.cuda.synchronize()

    # Per-phase timing (with sync for accuracy)
    transfer_ms = 0.0
    forward_ms = 0.0
    postproc_ms = 0.0
    copyback_ms = 0.0

    for _ in range(num_iters):
        # Transfer
        torch.cuda.synchronize()
        t0 = time.monotonic()
        b = batch_cpu.to(device)
        torch.cuda.synchronize()
        t1 = time.monotonic()
        transfer_ms += (t1 - t0) * 1000

        # Forward
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pl, wdl, _, _ = model(b)
        torch.cuda.synchronize()
        t2 = time.monotonic()
        forward_ms += (t2 - t1) * 1000

        # Softmax + WDL conversion
        policies = torch.softmax(pl.float(), dim=-1)
        wdl_probs = torch.softmax(wdl.float(), dim=-1)
        values = wdl_probs[:, 2] - wdl_probs[:, 0]
        torch.cuda.synchronize()
        t3 = time.monotonic()
        postproc_ms += (t3 - t2) * 1000

        # Copy back to CPU
        _ = policies.cpu().numpy()
        _ = values.cpu().numpy()
        torch.cuda.synchronize()
        t4 = time.monotonic()
        copyback_ms += (t4 - t3) * 1000

    return {
        "batch_size": batch_size,
        "transfer_ms": transfer_ms / num_iters,
        "forward_ms": forward_ms / num_iters,
        "postproc_ms": postproc_ms / num_iters,
        "copyback_ms": copyback_ms / num_iters,
        "total_ms": (transfer_ms + forward_ms + postproc_ms + copyback_ms) / num_iters,
        "positions_per_sec": batch_size / ((transfer_ms + forward_ms + postproc_ms + copyback_ms) / num_iters / 1000),
    }


@torch.no_grad()
def bench_cuda_graph(
    model: torch.nn.Module,
    batch_size: int,
    device: torch.device,
    num_warmup: int = 10,
    num_iters: int = 50,
) -> dict[str, float] | None:
    """Benchmark with CUDA graphs (eliminates kernel launch overhead)."""
    # CUDA graphs require fixed tensor addresses
    static_input = make_batch(batch_size, device)

    # Check if model is compatible with CUDA graphs
    try:
        # Warmup for graph capture
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _ = model(static_input)
        torch.cuda.synchronize()

        # Capture graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                static_pl, static_wdl, static_mat, static_act = model(static_input)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"  CUDA graph capture failed: {e}")
        return None

    # Warmup replay
    for _ in range(num_warmup):
        static_input.copy_(make_batch(batch_size, device))
        graph.replay()
    torch.cuda.synchronize()

    # Timed replay
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    per_iter_ms = total_ms / num_iters

    return {
        "batch_size": batch_size,
        "forward_ms": per_iter_ms,
        "per_position_us": per_iter_ms * 1000 / batch_size,
        "positions_per_sec": batch_size / (per_iter_ms / 1000),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone inference benchmark")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        sys.exit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.0f} GB)")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    batch_sizes = [1, 8, 32, 64, 128, 256, 512, 768, 1024, 2048, 4096]

    # ── Test 1: Compiled model, forward-only (data on GPU) ──
    print(f"\n{'='*70}")
    print("TEST 1: Forward pass only (data pre-loaded on GPU, torch.compile)")
    print(f"{'='*70}")

    model, config = setup_model(args.checkpoint, device, compile=not args.no_compile)

    print(f"\n{'Batch':>6} {'Forward':>10} {'Per-pos':>10} {'Pos/sec':>12}")
    print(f"{'─'*6} {'─'*10} {'─'*10} {'─'*12}")
    for bs in batch_sizes:
        batch = make_batch(bs, device)
        r = bench_forward(model, batch, num_iters=args.iters)
        print(f"{bs:>6} {r['forward_ms']:>9.2f}ms {r['per_position_us']:>8.1f}μs {r['positions_per_sec']:>11,.0f}")
        del batch
        torch.cuda.empty_cache()

    # ── Test 2: Full pipeline (CPU→GPU→forward→softmax→CPU) ──
    print(f"\n{'='*70}")
    print("TEST 2: Full evaluator pipeline (CPU→GPU→forward→softmax→CPU numpy)")
    print(f"{'='*70}")

    print(f"\n{'Batch':>6} {'Xfer':>8} {'Fwd':>8} {'Post':>8} {'Copy':>8} {'Total':>9} {'Pos/sec':>12}")
    print(f"{'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*9} {'─'*12}")
    for bs in [64, 128, 256, 512, 768, 1024]:
        batch_cpu = make_batch(bs, torch.device("cpu"))
        r = bench_full_pipeline(model, batch_cpu, device, num_iters=args.iters)
        print(
            f"{bs:>6} {r['transfer_ms']:>7.2f}ms {r['forward_ms']:>7.2f}ms "
            f"{r['postproc_ms']:>7.2f}ms {r['copyback_ms']:>7.2f}ms "
            f"{r['total_ms']:>8.2f}ms {r['positions_per_sec']:>11,.0f}"
        )
        del batch_cpu
        torch.cuda.empty_cache()

    # ── Test 3: Eager vs compiled ──
    print(f"\n{'='*70}")
    print("TEST 3: Eager mode (no torch.compile) — same batch sizes")
    print(f"{'='*70}")

    model_eager, _ = setup_model(args.checkpoint, device, compile=False)

    print(f"\n{'Batch':>6} {'Eager':>10} {'Compiled':>10} {'Speedup':>8}")
    print(f"{'─'*6} {'─'*10} {'─'*10} {'─'*8}")
    for bs in [64, 256, 512, 1024]:
        batch = make_batch(bs, device)
        r_eager = bench_forward(model_eager, batch, num_iters=args.iters)
        r_compiled = bench_forward(model, batch, num_iters=args.iters)
        speedup = r_eager["forward_ms"] / max(r_compiled["forward_ms"], 1e-6)
        print(
            f"{bs:>6} {r_eager['forward_ms']:>9.2f}ms {r_compiled['forward_ms']:>9.2f}ms {speedup:>7.2f}x"
        )
        del batch
        torch.cuda.empty_cache()

    del model_eager
    torch.cuda.empty_cache()

    # ── Test 4: CUDA Graphs ──
    print(f"\n{'='*70}")
    print("TEST 4: CUDA Graphs (eliminates kernel launch overhead)")
    print(f"{'='*70}")

    # Need uncompiled model for CUDA graphs (compile + graphs don't always mix)
    model_for_graphs, _ = setup_model(args.checkpoint, device, compile=False)

    print(f"\n{'Batch':>6} {'Graph':>10} {'Eager':>10} {'Speedup':>8}")
    print(f"{'─'*6} {'─'*10} {'─'*10} {'─'*8}")
    for bs in [64, 256, 512, 1024]:
        batch = make_batch(bs, device)
        r_eager = bench_forward(model_for_graphs, batch, num_iters=args.iters)
        r_graph = bench_cuda_graph(model_for_graphs, bs, device, num_iters=args.iters)
        if r_graph:
            speedup = r_eager["forward_ms"] / max(r_graph["forward_ms"], 1e-6)
            print(
                f"{bs:>6} {r_graph['forward_ms']:>9.2f}ms {r_eager['forward_ms']:>9.2f}ms {speedup:>7.2f}x"
            )
        else:
            print(f"{bs:>6} {'FAILED':>10} {r_eager['forward_ms']:>9.2f}ms {'N/A':>8}")
        del batch
        torch.cuda.empty_cache()

    # ── VRAM summary ──
    print(f"\n{'='*70}")
    print("VRAM USAGE")
    print(f"{'='*70}")
    for bs in [256, 512, 1024, 2048, 4096]:
        torch.cuda.reset_peak_memory_stats(device)
        batch = make_batch(bs, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _ = model(batch)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"  Batch {bs:>5}: {peak:.2f} GB")
        del batch
        torch.cuda.empty_cache()

    print(f"\nDone.")


if __name__ == "__main__":
    main()
