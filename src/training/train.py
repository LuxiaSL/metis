"""Main training loop for chess self-play.

Adapted from luxia-base training pipeline:
- DDP support (single-node multi-GPU)
- Muon + AdamW hybrid optimizer
- Warmup → constant LR schedule
- Self-play → train → eval loop
- Checkpoint management
- Wandb logging

Usage:
    # Single GPU
    python -m src.training.train --model_size medium

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=2 -m src.training.train --model_size medium
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from src.model.transformer import ChessModelConfig, ChessTransformer, MODEL_CONFIGS
from src.chess.board import NUM_PIECE_TYPES
from src.chess.self_play import SelfPlayConfig, SelfPlayWorker
from src.chess.evaluation import EvalConfig, StockfishEvaluator
from src.training.replay_buffer import ReplayBuffer
from src.training.muon import build_hybrid_optimizer, HybridScheduler
from src.monitoring.geometric import GeometricMonitor, MonitorConfig

logger = logging.getLogger(__name__)


# ── NCA Bootstrap ─────────────────────────────────────────────────────────


def run_nca_bootstrap(
    model: ChessTransformer,
    device: torch.device,
    num_rules: int = 2000,
    sims_per_rule: int = 8,
    max_pairs: int = 2_000_000,
    training_steps: int = 10000,
    batch_size: int = 256,
    lr: float = 3e-4,
) -> None:
    """NCA pre-training phase: teach attention circuits grid dynamics.

    Generates 8x8 NCA trajectories, trains the model to predict next-frame
    cell states, then reinitializes embeddings for chess.
    """
    from src.nca.generator import ChessNCAConfig, generate_nca_dataset

    logger.info("=" * 60)
    logger.info("NCA Bootstrap Phase")
    logger.info("=" * 60)

    # Generate NCA dataset
    nca_config = ChessNCAConfig()
    inputs, targets = generate_nca_dataset(
        config=nca_config,
        num_rules=num_rules,
        sims_per_rule=sims_per_rule,
        device=device,
        max_pairs=max_pairs,
    )
    logger.info("NCA dataset: %d frame pairs", len(inputs))

    # Temporary prediction head: predict next-frame cell states (13 classes per cell)
    nca_head = torch.nn.Linear(model.config.hidden_size, NUM_PIECE_TYPES).to(device)
    torch.nn.init.normal_(nca_head.weight, std=0.02)
    torch.nn.init.zeros_(nca_head.bias)

    # Simple AdamW for NCA phase (no Muon — this is short warmup)
    all_params = list(model.parameters()) + list(nca_head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=0.01)

    # Training loop
    model.train()
    n = len(inputs)
    for step in range(training_steps):
        # Random batch
        idx = torch.randint(0, n, (batch_size,))
        batch_inputs = inputs[idx].to(device)
        batch_targets = targets[idx].to(device)  # (B, 64) long

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            policy_logits, _ = model(batch_inputs)

            # Use the backbone hidden states for NCA prediction.
            # Re-run just the backbone to get hidden states (skip policy/value heads).
            # More efficient: use the square features from the policy head's input.
            # The policy head linear is (D, 73) — we need (D, 13) from nca_head.
            # Hack: get the pre-policy-head features by running forward and
            # extracting from the hook. Simpler: just run the model normally
            # and use a separate head on the square outputs.

            # Run backbone only
            B = batch_inputs.shape[0]
            castling = model.castling_embed(batch_inputs[:, 0])
            ep = model.ep_embed(batch_inputs[:, 1])
            side = model.side_embed(batch_inputs[:, 2])
            pieces = model.piece_embed(batch_inputs[:, 3:])
            pos_ids = torch.arange(model.config.seq_len, device=device)
            pos = model.pos_embed(pos_ids)
            x = torch.cat([
                (castling + pos[0]).unsqueeze(1),
                (ep + pos[1]).unsqueeze(1),
                (side + pos[2]).unsqueeze(1),
                pieces + pos[3:].unsqueeze(0),
            ], dim=1)
            for layer in model.layers:
                x = layer(x)
            x = model.norm(x)

            # Predict next cell states from square features
            square_features = x[:, model.config.num_global_tokens:, :]  # (B, 64, D)
            nca_logits = nca_head(square_features)  # (B, 64, 13)

            loss = F.cross_entropy(
                nca_logits.reshape(-1, NUM_PIECE_TYPES),
                batch_targets.reshape(-1),
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % 500 == 0:
            logger.info("NCA step %d/%d: loss=%.4f", step, training_steps, loss.item())

    # Transition to chess: reinitialize embeddings, discard NCA head
    logger.info("NCA bootstrap complete. Reinitializing embeddings for chess...")
    model.reinit_embeddings_for_chess()
    del nca_head, optimizer, inputs, targets
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    logger.info("=" * 60)


# ── Distributed helpers ────────────────────────────────────────────────────


def setup_distributed() -> tuple[int, int, int]:
    """Initialize DDP. Returns (rank, world_size, local_rank)."""
    if "RANK" not in os.environ:
        return 0, 1, 0  # Single GPU fallback

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


# ── Loss computation ──────────────────────────────────────────────────────


def compute_loss(
    policy_logits: torch.Tensor,
    value_pred: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
    z_loss_weight: float = 1e-5,
) -> dict[str, torch.Tensor]:
    """Compute AlphaZero-style training loss.

    Args:
        policy_logits: (B, 4672) raw policy logits from model.
        value_pred: (B, 1) tanh-squashed value predictions.
        target_policy: (B, 4672) MCTS visit count distributions (sums to 1).
        target_value: (B,) game outcomes from side-to-move perspective.
        z_loss_weight: z-loss coefficient for policy logit regularization.

    Returns:
        Dict with 'loss', 'policy_loss', 'value_loss', 'z_loss' tensors.
    """
    # Policy loss: cross-entropy between MCTS policy and model policy
    log_probs = F.log_softmax(policy_logits, dim=-1)
    policy_loss = -(target_policy * log_probs).sum(dim=-1).mean()

    # Value loss: MSE between predicted and actual outcome
    value_loss = F.mse_loss(value_pred.squeeze(-1), target_value)

    # z-loss: prevent policy logit explosion (from luxia-base)
    log_z = torch.logsumexp(policy_logits, dim=-1)
    z_loss = z_loss_weight * (log_z ** 2).mean()

    total_loss = policy_loss + value_loss + z_loss

    return {
        "loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "z_loss": z_loss,
    }


# ── Training loop ─────────────────────────────────────────────────────────


def train(args: argparse.Namespace) -> None:
    """Main training entry point."""
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    # ── Model ──────────────────────────────────────────────────────────
    model_kwargs = MODEL_CONFIGS.get(args.model_size, {})
    config = ChessModelConfig(**model_kwargs)

    if args.attn_res:
        config.attn_res = True
    if args.no_qk_norm:
        config.qk_norm = False

    model = ChessTransformer(config)

    if is_main_process():
        param_count = sum(p.numel() for p in model.parameters())
        logger.info("Model: %s (%s params)", args.model_size, f"{param_count:,}")
        logger.info("Config: %s", config)

    model = model.to(device)

    # Enable TF32 for matmuls
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── Optimizer ──────────────────────────────────────────────────────
    muon_opt, adamw_opt = build_hybrid_optimizer(
        model,
        muon_lr=args.muon_lr,
        muon_momentum=0.95,
        muon_weight_decay=args.muon_wd,
        muon_ns_iterations=5,
        muon_ns_coefficients=args.ns_coefficients,
        adamw_lr=args.adamw_lr,
        adamw_betas=(0.9, 0.95),
        adamw_weight_decay=0.1,
    )

    # Warmup-only schedule (constant LR after warmup)
    scheduler = HybridScheduler(
        muon_opt, adamw_opt,
        warmup_steps=args.warmup_steps,
        total_steps=999_999_999,   # Effectively no decay
        decay_start_pct=1.0,       # Never decay
    )

    # ── DDP ────────────────────────────────────────────────────────────
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)

    raw_model = model.module if isinstance(model, DDP) else model

    # ── Self-play config ───────────────────────────────────────────────
    self_play_config = SelfPlayConfig(
        num_parallel=args.num_parallel_games,
        mcts_simulations=args.mcts_simulations,
        temperature_threshold=args.temperature_threshold,
    )
    self_play_worker = SelfPlayWorker(
        model=raw_model, config=self_play_config, device=device,
    )

    # ── Replay buffer ─────────────────────────────────────────────────
    replay_buffer = ReplayBuffer(capacity=args.buffer_size)

    # ── Evaluation ────────────────────────────────────────────────────
    evaluator: Optional[StockfishEvaluator] = None
    if is_main_process() and args.eval_every > 0:
        eval_config = EvalConfig(
            stockfish_path=args.stockfish_path,
            depths=args.eval_depths,
            games_per_depth=args.eval_games_per_depth,
            mcts_simulations=args.eval_mcts_sims,
        )
        try:
            evaluator = StockfishEvaluator(config=eval_config, device=device)
        except RuntimeError as e:
            logger.warning("Stockfish evaluation disabled: %s", e)

    # ── Wandb ─────────────────────────────────────────────────────────
    wandb_run = None
    if is_main_process() and args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                config=vars(args),
            )
        except Exception as e:
            logger.warning("Wandb init failed: %s", e)

    # ── NCA Bootstrap (Phase 0) ──────────────────────────────────────
    if args.nca_bootstrap:
        if is_main_process():
            logger.info("Running NCA bootstrap phase...")
        run_nca_bootstrap(
            model=raw_model,
            device=device,
            num_rules=args.nca_num_rules,
            sims_per_rule=args.nca_sims_per_rule,
            max_pairs=args.nca_max_pairs,
            training_steps=args.nca_training_steps,
            batch_size=args.batch_size,
        )
        # Rebuild optimizers with fresh state after NCA
        muon_opt, adamw_opt = build_hybrid_optimizer(
            raw_model,
            muon_lr=args.muon_lr, muon_momentum=0.95,
            muon_weight_decay=args.muon_wd, muon_ns_iterations=5,
            muon_ns_coefficients=args.ns_coefficients,
            adamw_lr=args.adamw_lr, adamw_betas=(0.9, 0.95),
            adamw_weight_decay=0.1,
        )
        scheduler = HybridScheduler(
            muon_opt, adamw_opt,
            warmup_steps=args.warmup_steps,
            total_steps=999_999_999, decay_start_pct=1.0,
        )

    # ── Geometric Monitoring ──────────────────────────────────────────
    monitor: Optional[GeometricMonitor] = None
    if is_main_process() and args.monitor:
        monitor = GeometricMonitor(raw_model, MonitorConfig(
            tier1_every=args.monitor_tier1_every,
            tier2_every=args.monitor_tier2_every,
        ))

    # ── Checkpoint loading ────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_iteration = 0

    latest_ckpt = ckpt_dir / "latest.pt"
    if latest_ckpt.exists() and args.resume:
        if is_main_process():
            logger.info("Resuming from %s", latest_ckpt)
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        muon_opt.load_state_dict(ckpt["muon_opt"])
        adamw_opt.load_state_dict(ckpt["adamw_opt"])
        start_iteration = ckpt.get("iteration", 0)
        del ckpt

    # ── Main training loop ────────────────────────────────────────────
    global_train_step = start_iteration * args.train_steps_per_iter

    for iteration in range(start_iteration, args.num_iterations):
        iter_start = time.time()

        # ── Self-play phase ───────────────────────────────────────
        if is_main_process():
            logger.info("Iteration %d: generating %d games...", iteration, args.games_per_iter)

        raw_model.eval()
        games = self_play_worker.generate_games(args.games_per_iter)

        total_positions = sum(len(g) for g in games)
        for game in games:
            replay_buffer.add_game(game)

        if is_main_process():
            logger.info(
                "Generated %d games, %d positions (buffer: %d)",
                len(games), total_positions, len(replay_buffer),
            )

        # Set probe batch for monitoring (once, from first generation)
        if monitor is not None and monitor._probe_batch is None and len(replay_buffer) >= 64:
            probe_boards, _, _ = replay_buffer.sample(64)
            monitor.set_probe_batch(probe_boards)

        # ── Training phase ────────────────────────────────────────
        if len(replay_buffer) < args.batch_size:
            if is_main_process():
                logger.info("Buffer too small (%d), skipping training", len(replay_buffer))
            continue

        model.train()
        train_metrics: dict[str, float] = {
            "loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "z_loss": 0.0,
        }

        for step in range(args.train_steps_per_iter):
            boards, target_policies, target_values = replay_buffer.sample(args.batch_size)
            boards = boards.to(device)
            target_policies = target_policies.to(device)
            target_values = target_values.to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                policy_logits, value_pred = model(boards)
                losses = compute_loss(
                    policy_logits, value_pred,
                    target_policies, target_values,
                    z_loss_weight=config.z_loss_weight,
                )

            losses["loss"].backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip,
            )

            muon_opt.step()
            adamw_opt.step()
            muon_opt.zero_grad(set_to_none=True)
            adamw_opt.zero_grad(set_to_none=True)

            global_train_step += 1
            scheduler.step(global_train_step)

            # Accumulate metrics
            for k in train_metrics:
                train_metrics[k] += losses[k].item()

        # Average metrics
        num_steps = args.train_steps_per_iter
        for k in train_metrics:
            train_metrics[k] /= num_steps

        iter_time = time.time() - iter_start

        # ── Logging ───────────────────────────────────────────────
        if is_main_process():
            lrs = scheduler.get_last_lr()
            logger.info(
                "Iter %d: loss=%.4f (policy=%.4f value=%.4f z=%.6f) "
                "lr_muon=%.2e lr_adam=%.2e grad=%.2f time=%.1fs",
                iteration,
                train_metrics["loss"],
                train_metrics["policy_loss"],
                train_metrics["value_loss"],
                train_metrics["z_loss"],
                lrs["muon_lr"],
                lrs["adamw_lr"],
                grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                iter_time,
            )

            if wandb_run is not None:
                log_dict = {
                    f"train/{k}": v for k, v in train_metrics.items()
                }
                log_dict["train/grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                log_dict["train/muon_lr"] = lrs["muon_lr"]
                log_dict["train/adamw_lr"] = lrs["adamw_lr"]
                log_dict["train/buffer_size"] = len(replay_buffer)
                log_dict["train/positions_generated"] = total_positions
                log_dict["train/iter_time"] = iter_time
                wandb_run.log(log_dict, step=iteration)

        # ── Geometric monitoring ──────────────────────────────────
        if monitor is not None and is_main_process():
            if global_train_step % args.monitor_tier1_every == 0:
                raw_model.eval()
                # Get policy logits for policy entropy metric
                probe = monitor._probe_batch
                if probe is not None:
                    with torch.no_grad():
                        p_logits, _ = raw_model(probe.to(device))
                    geo_metrics = monitor.tier1(global_train_step, policy_logits=p_logits)
                    if wandb_run is not None:
                        wandb_run.log(geo_metrics, step=iteration)

            if global_train_step % args.monitor_tier2_every == 0:
                raw_model.eval()
                geo_metrics = monitor.tier2(global_train_step)
                if wandb_run is not None:
                    wandb_run.log(geo_metrics, step=iteration)

        # ── Evaluation ────────────────────────────────────────────
        if (
            evaluator is not None
            and args.eval_every > 0
            and (iteration + 1) % args.eval_every == 0
        ):
            if is_main_process():
                logger.info("Running Stockfish evaluation...")
                raw_model.eval()
                eval_results = evaluator.evaluate(raw_model)

                if wandb_run is not None:
                    wandb_run.log(eval_results, step=iteration)

        # ── Checkpoint ────────────────────────────────────────────
        if (
            is_main_process()
            and args.save_every > 0
            and (iteration + 1) % args.save_every == 0
        ):
            ckpt_path = ckpt_dir / f"iter_{iteration:06d}.pt"
            torch.save({
                "model": raw_model.state_dict(),
                "muon_opt": muon_opt.state_dict(),
                "adamw_opt": adamw_opt.state_dict(),
                "iteration": iteration + 1,
                "config": vars(config) if hasattr(config, '__dict__') else str(config),
            }, ckpt_path)
            # Symlink latest
            if latest_ckpt.exists() or latest_ckpt.is_symlink():
                latest_ckpt.unlink()
            latest_ckpt.symlink_to(ckpt_path.name)
            logger.info("Saved checkpoint: %s", ckpt_path)

    # ── Cleanup ───────────────────────────────────────────────────────
    if evaluator is not None:
        evaluator.close()
    if wandb_run is not None:
        wandb_run.finish()
    cleanup_distributed()


# ── CLI ───────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chess engine training")

    # Model
    parser.add_argument("--model_size", type=str, default="medium",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--attn_res", action="store_true", help="Enable Block Attention Residuals")
    parser.add_argument("--no_qk_norm", action="store_true", help="Disable QK-norm")

    # Training
    parser.add_argument("--num_iterations", type=int, default=1000)
    parser.add_argument("--games_per_iter", type=int, default=256)
    parser.add_argument("--train_steps_per_iter", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--buffer_size", type=int, default=1_000_000)

    # Optimizer
    parser.add_argument("--muon_lr", type=float, default=0.02)
    parser.add_argument("--muon_wd", type=float, default=0.01)
    parser.add_argument("--adamw_lr", type=float, default=3e-4)
    parser.add_argument("--ns_coefficients", type=str, default="gram_ns")
    parser.add_argument("--warmup_steps", type=int, default=500)

    # Self-play
    parser.add_argument("--num_parallel_games", type=int, default=64)
    parser.add_argument("--mcts_simulations", type=int, default=800)
    parser.add_argument("--temperature_threshold", type=int, default=30)

    # Evaluation
    parser.add_argument("--eval_every", type=int, default=10,
                        help="Evaluate every N iterations (0 to disable)")
    parser.add_argument("--eval_depths", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--eval_games_per_depth", type=int, default=20)
    parser.add_argument("--eval_mcts_sims", type=int, default=400)
    parser.add_argument("--stockfish_path", type=str, default="stockfish")

    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")

    # NCA Bootstrap
    parser.add_argument("--nca_bootstrap", action="store_true",
                        help="Run NCA pre-training before self-play")
    parser.add_argument("--nca_num_rules", type=int, default=2000)
    parser.add_argument("--nca_sims_per_rule", type=int, default=8)
    parser.add_argument("--nca_max_pairs", type=int, default=2_000_000)
    parser.add_argument("--nca_training_steps", type=int, default=10000)

    # Monitoring
    parser.add_argument("--monitor", action="store_true",
                        help="Enable geometric health monitoring")
    parser.add_argument("--monitor_tier1_every", type=int, default=500)
    parser.add_argument("--monitor_tier2_every", type=int, default=5000)

    # Logging
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="chess-engine")
    parser.add_argument("--wandb_name", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
