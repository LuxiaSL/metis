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
import gc
import logging
import os
import signal
import threading
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from src.model.transformer import ChessModelConfig, ChessTransformer, MODEL_CONFIGS
from src.chess.board import NUM_PIECE_TYPES
from src.chess.self_play import SelfPlayConfig, SelfPlayWorker, ParallelSelfPlay
from src.chess.evaluation import EvalConfig, StockfishEvaluator
from src.training.replay_buffer import ReplayBuffer
from src.training.muon import build_hybrid_optimizer, HybridScheduler
from src.monitoring.geometric import GeometricMonitor, MonitorConfig

logger = logging.getLogger(__name__)


# ── Graceful shutdown ──────────────────────────────────────────────────────

_shutdown_requested = False
_shutdown_event = threading.Event()


def _request_shutdown(signum: int, frame: object) -> None:
    """Signal handler for graceful exit. Second signal forces immediate exit."""
    global _shutdown_requested
    if _shutdown_requested:
        logger.warning("Second signal received — forcing exit")
        raise SystemExit(1)
    _shutdown_requested = True
    _shutdown_event.set()
    logger.info(
        "Shutdown requested (signal %d) — saving checkpoint and exiting",
        signum,
    )


# ── NCA Bootstrap ─────────────────────────────────────────────────────────


def run_nca_bootstrap(
    model: ChessTransformer,
    device: torch.device,
    checkpoint_dir: Path,
    num_rules: int = 2000,
    sims_per_rule: int = 8,
    max_pairs: int = 2_000_000,
    min_steps: int = 2000,
    max_steps: int = 20000,
    batch_size: int = 256,
    muon_lr: float = 0.02,
    adamw_lr: float = 3e-4,
    ns_coefficients: str = "gram_ns",
    seed: int = 17,
    saturation_check_every: int = 200,
    saturation_threshold: float = 0.3,
    save_every: int = 1000,
    dataset_path: Optional[str] = None,
    skip_reinit: bool = False,
    wandb_run: object = None,
) -> None:
    """NCA pre-training: teach attention circuits 8x8 grid dynamics.

    Uses Muon + AdamW (same as chess training). Monitors geometric metrics
    for saturation and stops when plateaued. Saves a checkpoint of the raw
    NCA-trained weights BEFORE reinitializing embeddings.

    The checkpoint allows:
    - Resuming without re-running NCA
    - Inspecting NCA-trained weights for analysis
    - Extracting geometric profiles for AttnRes boundary selection
    """
    import random as _random
    from src.nca.generator import ChessNCAConfig, generate_nca_dataset

    logger.info("=" * 60)
    logger.info("NCA Bootstrap Phase (seed=%d)", seed)
    logger.info("=" * 60)

    # Check for existing NCA checkpoint (skip if found)
    nca_ckpt_path = checkpoint_dir / "nca_checkpoint.pt"
    nca_resume_path = checkpoint_dir / "nca_resume.pt"

    if nca_ckpt_path.exists():
        logger.info("Found final NCA checkpoint at %s — skipping NCA phase", nca_ckpt_path)
        ckpt = torch.load(nca_ckpt_path, map_location=device, weights_only=False)
        # Filter out keys with shape mismatches (e.g. old scalar value_fc2 vs new WDL)
        ckpt_state = ckpt["model"]
        model_state = model.state_dict()
        filtered = {
            k: v for k, v in ckpt_state.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        skipped = set(ckpt_state.keys()) - set(filtered.keys())
        if skipped:
            logger.info("Skipped %d keys with shape mismatch: %s", len(skipped), list(skipped)[:10])
        model.load_state_dict(filtered, strict=False)
        logger.info("Loaded NCA weights (step %d, loss %.4f)", ckpt.get("step", -1), ckpt.get("loss", -1))
        if not skip_reinit:
            model.reinit_embeddings_for_chess()
            logger.info("Embeddings reinitialized for chess.")
        del ckpt
        return

    # Resume state (populated if intermediate checkpoint exists)
    resume_data: Optional[dict] = None
    start_step = 0

    if nca_resume_path.exists():
        logger.info("Resuming NCA from intermediate checkpoint: %s", nca_resume_path)
        resume_data = torch.load(nca_resume_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_data["model"], strict=False)
        start_step = resume_data["step"] + 1
        logger.info(
            "Will resume from step %d (loss %.4f, best %.4f)",
            start_step, resume_data.get("loss", -1), resume_data.get("best_loss", -1),
        )

    # Seed for reproducibility (deterministic dataset on fresh start or resume)
    torch.manual_seed(seed)
    _random.seed(seed)

    # Load or generate NCA dataset
    if dataset_path is not None:
        logger.info("Loading NCA dataset from %s", dataset_path)
        ds = torch.load(dataset_path, map_location="cpu", weights_only=False)
        inputs = ds["inputs"].long()  # uint8 → int64 for embedding lookup
        targets = ds["targets"].long()
        if max_pairs is not None and len(inputs) > max_pairs:
            inputs = inputs[:max_pairs]
            targets = targets[:max_pairs]
        del ds
    else:
        nca_config = ChessNCAConfig()
        inputs, targets = generate_nca_dataset(
            config=nca_config,
            num_rules=num_rules,
            sims_per_rule=sims_per_rule,
            device=device,
            max_pairs=max_pairs,
        )
    # Split: 90% train, 10% held-out eval (same seed — avoids cross-seed confound)
    n_total = len(inputs)
    perm = torch.randperm(n_total)
    n_eval = max(1000, n_total // 10)
    eval_idx, train_idx = perm[:n_eval], perm[n_eval:]
    eval_inputs, eval_targets = inputs[eval_idx], targets[eval_idx]
    inputs, targets = inputs[train_idx], targets[train_idx]
    n = len(inputs)
    logger.info(
        "NCA dataset: %d train + %d eval pairs (%.1fM + %.1fM tokens)",
        n, n_eval, n * 67 / 1e6, n_eval * 67 / 1e6,
    )

    # NCA prediction head: per-square → 13 cell states
    nca_head = torch.nn.Linear(model.config.hidden_size, NUM_PIECE_TYPES).to(device)
    torch.nn.init.normal_(nca_head.weight, std=0.02)
    torch.nn.init.zeros_(nca_head.bias)

    # Muon + AdamW (same hybrid strategy as chess training)
    muon_opt, adamw_opt = build_hybrid_optimizer(
        model,
        muon_lr=muon_lr,
        muon_momentum=0.95,
        muon_weight_decay=0.01,
        muon_ns_iterations=5,
        muon_ns_coefficients=ns_coefficients,
        adamw_lr=adamw_lr,
        adamw_betas=(0.9, 0.95),
        adamw_weight_decay=0.1,
    )
    # NCA head goes to AdamW (it's a temporary head, not a 2D weight matrix for Muon)
    nca_head_opt = torch.optim.AdamW(nca_head.parameters(), lr=adamw_lr, weight_decay=0.01)

    # Warmup schedule
    warmup_steps = min(500, min_steps // 4)

    # Geometric monitoring for saturation detection
    monitor = GeometricMonitor(model, MonitorConfig(tier1_every=saturation_check_every))
    # Use a fixed probe batch for longitudinal tracking
    probe_idx = torch.randperm(n)[:64]
    monitor.set_probe_batch(inputs[probe_idx])

    # Saturation tracking: store metric history for slope ratio computation
    metric_history: dict[str, list[float]] = {}

    def _check_saturation(step: int) -> bool:
        """Check if key metrics have plateaued (slope ratio < threshold).

        Slope ratio = |change in last third of history| / |change in first third|.
        When < threshold (0.3), the metric is changing <30% as fast as early training.
        ALL required metrics must be saturated simultaneously.
        """
        if step < min_steps:
            return False
        # Core metrics: loss must plateau, plus at least 2 geometric metrics
        required = ["loss", "geo/rankme_last"]
        geometric = [
            k for k in metric_history
            if k.startswith("geo/layer_") and k.endswith("/anisotropy")
        ]
        # Add the first available anisotropy metric
        if geometric:
            required.append(geometric[0])

        saturated_count = 0
        for key in required:
            if key not in metric_history or len(metric_history[key]) < 6:
                return False
            vals = metric_history[key]
            third = len(vals) // 3
            if third < 2:
                return False
            first_slope = abs(vals[third - 1] - vals[0])
            last_slope = abs(vals[-1] - vals[-third])
            first_slope = max(first_slope, 1e-8)
            ratio = last_slope / first_slope
            if ratio <= saturation_threshold:
                saturated_count += 1
            if wandb_run is not None:
                wandb_run.log({f"nca/saturation_ratio/{key}": ratio}, step=step)

        if saturated_count == len(required):
            logger.info(
                "NCA saturation detected at step %d (%d/%d metrics plateaued)",
                step, saturated_count, len(required),
            )
            return True
        return False

    # ── Training loop ─────────────────────────────────────────────
    model.train()
    autocast_enabled = device.type == "cuda"
    best_loss = float("inf")

    # Apply resume state (optimizer + head weights, metric history)
    if resume_data is not None:
        nca_head.load_state_dict(resume_data["nca_head"])
        muon_opt.load_state_dict(resume_data["muon_opt"])
        adamw_opt.load_state_dict(resume_data["adamw_opt"])
        nca_head_opt.load_state_dict(resume_data["nca_head_opt"])
        best_loss = resume_data.get("best_loss", float("inf"))
        metric_history.update(resume_data.get("metric_history", {}))
        logger.info("Loaded resume state (best_loss=%.4f, resuming at step %d)", best_loss, start_step)
        del resume_data

    for step in range(start_step, max_steps):
        # LR warmup
        if step < warmup_steps:
            lr_mult = (step + 1) / warmup_steps
            for g in muon_opt.param_groups:
                g["lr"] = muon_lr * lr_mult
            for g in adamw_opt.param_groups:
                g["lr"] = adamw_lr * lr_mult
            for g in nca_head_opt.param_groups:
                g["lr"] = adamw_lr * lr_mult

        idx = torch.randint(0, n, (batch_size,))
        batch_inputs = inputs[idx].to(device)
        batch_targets = targets[idx].to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            # Use backbone_forward (no policy/value heads)
            x = model.backbone_forward(batch_inputs)
            square_features = x[:, model.config.num_global_tokens:, :]
            nca_logits = nca_head(square_features)

            loss = F.cross_entropy(
                nca_logits.reshape(-1, NUM_PIECE_TYPES),
                batch_targets.reshape(-1),
            )

        loss.backward()
        all_params = list(model.parameters()) + list(nca_head.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(all_params, 1.0).item()

        # Per-component grad norms (before step clears grads)
        model_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).item()
        head_grad_norm = torch.nn.utils.clip_grad_norm_(nca_head.parameters(), float("inf")).item()

        muon_opt.step()
        adamw_opt.step()
        nca_head_opt.step()
        muon_opt.zero_grad(set_to_none=True)
        adamw_opt.zero_grad(set_to_none=True)
        nca_head_opt.zero_grad(set_to_none=True)

        loss_val = loss.item()
        best_loss = min(best_loss, loss_val)

        # Logging + saturation check
        if step % saturation_check_every == 0:
            metric_history.setdefault("loss", []).append(loss_val)
            metric_history.setdefault("grad_norm", []).append(grad_norm)
            metric_history.setdefault("model_grad_norm", []).append(model_grad_norm)
            metric_history.setdefault("head_grad_norm", []).append(head_grad_norm)

            # Held-out eval (same-seed, avoids cross-seed confound)
            model.eval()
            with torch.no_grad():
                eval_batch = eval_inputs[:min(2048, len(eval_inputs))].to(device)
                eval_tgt = eval_targets[:min(2048, len(eval_targets))].to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                    eval_x = model.backbone_forward(eval_batch)
                    eval_sq = eval_x[:, model.config.num_global_tokens:, :]
                    eval_logits = nca_head(eval_sq)
                    eval_loss = F.cross_entropy(
                        eval_logits.reshape(-1, NUM_PIECE_TYPES),
                        eval_tgt.reshape(-1),
                    ).item()
            metric_history.setdefault("eval_loss", []).append(eval_loss)

            # Tier 1 geometric metrics
            geo_metrics = monitor.tier1(step, probe_batch=monitor._probe_batch)
            model.train()

            for k, v in geo_metrics.items():
                metric_history.setdefault(k, []).append(v)

            logger.info(
                "NCA step %d/%d: loss=%.4f eval=%.4f (best=%.4f) RankMe=%.1f",
                step, max_steps, loss_val, eval_loss, best_loss,
                geo_metrics.get("geo/rankme_last", 0),
            )

            # Log to wandb
            if wandb_run is not None:
                log_dict = {
                    "nca/loss": loss_val,
                    "nca/eval_loss": eval_loss,
                    "nca/best_loss": best_loss,
                    "nca/muon_lr": muon_opt.param_groups[0]["lr"],
                    "nca/adamw_lr": adamw_opt.param_groups[0]["lr"],
                    "nca/epoch": step * batch_size / n,
                    # Gradient health
                    "nca/grad_norm": grad_norm,
                    "nca/model_grad_norm": model_grad_norm,
                    "nca/head_grad_norm": head_grad_norm,
                }
                # Weight norms per layer (Frobenius)
                for li, layer in enumerate(model.layers):
                    q_norm = layer.attn.q_proj.weight.float().norm().item()
                    o_norm = layer.attn.o_proj.weight.float().norm().item()
                    gate_norm = layer.ffn.gate_proj.weight.float().norm().item()
                    log_dict[f"nca/weight_norm/layer_{li}/q_proj"] = q_norm
                    log_dict[f"nca/weight_norm/layer_{li}/o_proj"] = o_norm
                    log_dict[f"nca/weight_norm/layer_{li}/gate_proj"] = gate_norm
                log_dict.update({f"nca/{k}": v for k, v in geo_metrics.items()})
                wandb_run.log(log_dict, step=step)

            # Check saturation after min_steps
            if _check_saturation(step):
                logger.info("Stopping NCA at step %d (saturation)", step)
                break

        # Intermediate save (periodic or on shutdown request)
        _should_save = (
            (save_every > 0 and step > 0 and step % save_every == 0)
            or _shutdown_requested
        )
        if _should_save:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ckpt_data = {
                "model": model.state_dict(),
                "nca_head": nca_head.state_dict(),
                "muon_opt": muon_opt.state_dict(),
                "adamw_opt": adamw_opt.state_dict(),
                "nca_head_opt": nca_head_opt.state_dict(),
                "step": step,
                "loss": loss_val,
                "best_loss": best_loss,
                "metric_history": metric_history,
                "seed": seed,
            }
            # Resume checkpoint (overwritten each time)
            torch.save(ckpt_data, nca_resume_path)
            # Named snapshot (kept for post-hoc analysis / boundary discovery)
            snapshot_path = checkpoint_dir / f"nca_step_{step:06d}.pt"
            torch.save(ckpt_data, snapshot_path)
            logger.info("Saved NCA checkpoint: step %d (%s)", step, snapshot_path.name)
            del ckpt_data

            if _shutdown_requested:
                logger.info("NCA interrupted at step %d — will resume on restart", step)
                return

    # ── Save NCA checkpoint BEFORE reinit ─────────────────────────
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "nca_head": nca_head.state_dict(),
        "step": step,
        "loss": loss_val,
        "best_loss": best_loss,
        "metric_history": metric_history,
        "seed": seed,
    }, nca_ckpt_path)
    logger.info("Saved NCA checkpoint: %s (step %d, loss %.4f)", nca_ckpt_path, step, loss_val)

    # Clean up intermediate checkpoint (no longer needed)
    if nca_resume_path.exists():
        nca_resume_path.unlink()

    # ── Transition to chess ───────────────────────────────────────
    if not skip_reinit:
        model.reinit_embeddings_for_chess()
        logger.info("Embeddings reinitialized for chess.")
    else:
        logger.info("skip_reinit=True — keeping raw NCA weights (for boundary analysis)")

    del nca_head, muon_opt, adamw_opt, nca_head_opt, inputs, targets
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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


def _scalar_to_soft_wdl(v: torch.Tensor) -> torch.Tensor:
    """Convert scalar value in [-1, 1] to soft WDL distribution.

    Maps v to (p_loss, p_draw, p_win) where:
        p_win  = max(0, v)
        p_loss = max(0, -v)
        p_draw = 1 - |v|

    Args:
        v: (B,) scalar values in [-1, 1].

    Returns:
        (B, 3) soft WDL distributions [loss, draw, win].
    """
    p_win = v.clamp(min=0.0)
    p_loss = (-v).clamp(min=0.0)
    p_draw = (1.0 - v.abs()).clamp(min=0.0)
    return torch.stack([p_loss, p_draw, p_win], dim=-1)


def compute_loss(
    policy_logits: torch.Tensor,
    wdl_logits: torch.Tensor,
    material_pred: torch.Tensor,
    activity_pred: torch.Tensor,
    moves_left_pred: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
    target_q_value: torch.Tensor,
    target_material: torch.Tensor,
    target_activity: torch.Tensor,
    target_moves_left: torch.Tensor,
    z_loss_weight: float = 1e-5,
    material_loss_weight: float = 0.1,
    activity_loss_weight: float = 0.05,
    moves_left_loss_weight: float = 0.02,
    q_blend: float = 0.0,
    sample_weights: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Compute training loss with WDL value head and auxiliary targets.

    Args:
        policy_logits: (B, 4672) raw policy logits from model.
        wdl_logits: (B, 3) win/draw/loss logits from model.
        material_pred: (B, 1) predicted material balance.
        activity_pred: (B, 1) predicted legal move count.
        moves_left_pred: (B, 1) predicted remaining game length.
        target_policy: (B, 4672) MCTS visit count distributions (sums to 1).
        target_value: (B,) z-targets — game outcomes {-1, 0, +1} from side-to-move.
        target_q_value: (B,) q-targets — MCTS root values from side-to-move.
        target_material: (B,) normalized material balance.
        target_activity: (B,) normalized legal move count.
        target_moves_left: (B,) normalized remaining game length.
        z_loss_weight: z-loss coefficient for policy logit regularization.
        material_loss_weight: weight for material prediction auxiliary loss.
        activity_loss_weight: weight for activity prediction auxiliary loss.
        moves_left_loss_weight: weight for moves-left prediction auxiliary loss.
        q_blend: blend factor for value target — 0.0 = pure z, 1.0 = pure q.
        sample_weights: (B,) optional per-sample weights from policy surprise.

    Returns:
        Dict with 'loss', 'policy_loss', 'value_loss', 'z_loss',
        'material_loss', 'activity_loss', 'moves_left_loss' tensors.
    """
    # Policy loss: cross-entropy between MCTS policy and model policy
    log_probs = F.log_softmax(policy_logits, dim=-1)
    per_sample_policy = -(target_policy * log_probs).sum(dim=-1)  # (B,)

    # WDL value loss: blend z-target (game outcome) and q-target (MCTS root value)
    if q_blend > 0.0:
        z_wdl = _scalar_to_soft_wdl(target_value)
        q_wdl = _scalar_to_soft_wdl(target_q_value)
        target_wdl = (1.0 - q_blend) * z_wdl + q_blend * q_wdl
        wdl_log_probs = F.log_softmax(wdl_logits, dim=-1)
        per_sample_value = -(target_wdl * wdl_log_probs).sum(dim=-1)
    else:
        target_class = (target_value + 1).long()
        per_sample_value = F.cross_entropy(wdl_logits, target_class, reduction="none")

    # z-loss: prevent policy logit explosion
    log_z = torch.logsumexp(policy_logits, dim=-1)
    per_sample_z = z_loss_weight * (log_z ** 2)

    # Auxiliary losses (per-sample)
    per_sample_material = (material_pred.squeeze(-1) - target_material) ** 2
    per_sample_activity = (activity_pred.squeeze(-1) - target_activity) ** 2
    per_sample_moves_left = (moves_left_pred.squeeze(-1) - target_moves_left) ** 2

    per_sample_loss = (
        per_sample_policy
        + per_sample_value
        + per_sample_z
        + material_loss_weight * per_sample_material
        + activity_loss_weight * per_sample_activity
        + moves_left_loss_weight * per_sample_moves_left
    )

    if sample_weights is not None:
        total_loss = (per_sample_loss * sample_weights).mean()
    else:
        total_loss = per_sample_loss.mean()

    # Report unweighted component means for logging consistency
    return {
        "loss": total_loss,
        "policy_loss": per_sample_policy.mean(),
        "value_loss": per_sample_value.mean(),
        "z_loss": per_sample_z.mean(),
        "material_loss": per_sample_material.mean(),
        "activity_loss": per_sample_activity.mean(),
        "moves_left_loss": per_sample_moves_left.mean(),
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

    # Register signal handlers for graceful shutdown (checkpoint on SIGTERM)
    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)

    # ── Model ──────────────────────────────────────────────────────────
    model_kwargs = MODEL_CONFIGS.get(args.model_size, {})
    config = ChessModelConfig(**model_kwargs)

    if args.attn_res:
        config.attn_res = True
    if args.attn_res_boundaries:
        config.attn_res = True
        config.attn_res_boundaries = [int(x) for x in args.attn_res_boundaries.split(",")]
    if args.no_qk_norm:
        config.qk_norm = False
    if args.activation_checkpointing:
        config.activation_checkpointing = True

    model = ChessTransformer(config)

    if is_main_process():
        param_count = sum(p.numel() for p in model.parameters())
        logger.info("Model: %s (%s params)", args.model_size, f"{param_count:,}")
        logger.info("Config: %s", config)

    model = model.to(device)

    # Enable TF32 for matmuls
    torch.backends.cuda.matmul.allow_tf32 = True

    # torch.compile is applied after checkpoint loading (below) to avoid
    # _orig_mod. key prefix mismatch when loading non-compiled checkpoints.
    torch.backends.cudnn.allow_tf32 = True

    # ── Checkpoint dir (needed by NCA bootstrap) ──────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Wandb (init early so NCA phase can log too) ────────────────────
    wandb_run = None
    if is_main_process() and args.wandb:
        try:
            import wandb
            if args.wandb_api_key:
                os.environ["WANDB_API_KEY"] = args.wandb_api_key
            # Use a deterministic run ID derived from wandb_name so that
            # resumed runs append to the same wandb run instead of creating
            # a new one (which resets the step counter).
            _wandb_id = args.wandb_name or "metis"
            wandb_run = wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=args.wandb_name,
                id=_wandb_id,
                resume="allow",
                config=vars(args),
            )
            # Use iteration as x-axis for all training metrics
            wandb.define_metric("iteration")
            wandb.define_metric("train/*", step_metric="iteration")
            wandb.define_metric("selfplay/*", step_metric="iteration")
            wandb.define_metric("cumulative/*", step_metric="iteration")
            wandb.define_metric("time/*", step_metric="iteration")
            wandb.define_metric("eval/*", step_metric="iteration")
            wandb.define_metric("geo/*", step_metric="iteration")
        except Exception as e:
            logger.warning("Wandb init failed: %s", e)

    # ── NCA Bootstrap (Phase 0, before DDP/optimizer) ─────────────────
    if args.nca_bootstrap:
        if is_main_process():
            logger.info("Running NCA bootstrap phase...")
        run_nca_bootstrap(
            model=model,
            device=device,
            checkpoint_dir=ckpt_dir,
            num_rules=args.nca_num_rules,
            sims_per_rule=args.nca_sims_per_rule,
            max_pairs=args.nca_max_pairs,
            min_steps=args.nca_min_steps,
            max_steps=args.nca_max_steps,
            batch_size=args.batch_size,
            muon_lr=args.muon_lr,
            adamw_lr=args.adamw_lr,
            ns_coefficients=args.ns_coefficients,
            seed=args.nca_seed,
            saturation_threshold=0.0 if args.nca_no_auto_stop else 0.3,
            save_every=args.nca_save_every,
            dataset_path=args.nca_dataset,
            skip_reinit=args.nca_skip_reinit,
            wandb_run=wandb_run,
        )
        if _shutdown_requested:
            logger.info("Shutdown during NCA bootstrap — exiting cleanly")
            cleanup_distributed()
            return

    # ── Optimizer (built on post-NCA model) ───────────────────────────
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
        num_workers=args.num_workers,
        mcts_simulations=args.mcts_simulations,
        temperature_threshold=args.temperature_threshold,
        dirichlet_epsilon=args.dirichlet_epsilon,
        num_virtual_leaves=args.num_virtual_leaves,
        mcts_algorithm=args.mcts_algorithm,
        gumbel_K=args.gumbel_K,
        gumbel_c_visit=args.gumbel_c_visit,
        playout_cap_fraction=args.playout_cap_fraction,
        fast_move_sims=args.fast_move_sims,
        material_adjudication_threshold=args.material_adjudication_threshold,
    )

    # Use multiprocessing when >1 worker, fallback to single-process.
    # Pass `model` (not raw_model) so torch.compile benefits apply to inference.
    # raw_model is kept for checkpoint saving (clean state_dict keys).
    if args.num_workers > 1:
        self_play_engine: ParallelSelfPlay | SelfPlayWorker = ParallelSelfPlay(
            model=model, config=self_play_config, device=device,
        )
    else:
        self_play_engine = SelfPlayWorker(
            model=model, config=self_play_config, device=device,
        )

    # ── Replay buffer ─────────────────────────────────────────────────
    replay_buffer = ReplayBuffer(capacity=args.buffer_size, decisive_boost=args.decisive_boost)

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

    # ── Geometric Monitoring ──────────────────────────────────────────
    monitor: Optional[GeometricMonitor] = None
    if is_main_process() and args.monitor:
        monitor = GeometricMonitor(raw_model, MonitorConfig(
            tier1_every=args.monitor_tier1_every,
            tier2_every=args.monitor_tier2_every,
        ))

    # ── Checkpoint loading ────────────────────────────────────────────
    start_iteration = 0

    latest_ckpt = ckpt_dir / "latest.pt"
    if latest_ckpt.exists() and args.resume:
        if is_main_process():
            logger.info("Resuming from %s", latest_ckpt)
        ckpt = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        # Filter shape mismatches for cross-version checkpoint compatibility
        ckpt_state = ckpt["model"]
        model_state = raw_model.state_dict()
        filtered = {
            k: v for k, v in ckpt_state.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        shape_skipped = set(ckpt_state.keys()) - set(filtered.keys())
        if shape_skipped:
            logger.warning("Skipped %d keys with shape mismatch: %s", len(shape_skipped), list(shape_skipped)[:10])
        missing, unexpected = raw_model.load_state_dict(filtered, strict=False)
        if missing:
            logger.info("Missing keys on resume (new params, using init): %s", missing[:10])
        if unexpected:
            logger.warning("Unexpected keys on resume (removed params): %s", unexpected[:10])
        muon_opt.load_state_dict(ckpt["muon_opt"])
        adamw_opt.load_state_dict(ckpt["adamw_opt"])
        start_iteration = ckpt.get("iteration", 0)
        if "replay_buffer" in ckpt:
            replay_buffer.load_state_dict(ckpt["replay_buffer"])
            logger.info("Restored replay buffer (%d positions)", len(replay_buffer))
        del ckpt

    # ── torch.compile (after checkpoint load to avoid key prefix mismatch) ──
    if args.compile:
        if is_main_process():
            logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        # Update self-play and eval to use compiled model for inference
        self_play_engine.model = model
        if is_main_process():
            logger.info("Compilation registered (will compile on first forward)")

    # ── Main training loop ────────────────────────────────────────────
    global_train_step = start_iteration * args.train_steps_per_iter

    # Cumulative counters (across all iterations in this run)
    cumul_games = 0
    cumul_positions = 0
    cumul_evals = 0
    # Rolling window for win rate trend (last 5 iterations)
    recent_win_rates: list[float] = []
    _WIN_RATE_WINDOW = 5

    for iteration in range(start_iteration, args.num_iterations):
        iter_start = time.time()
        iter_log_dict: dict[str, float] = {}

        # ── Self-play phase ───────────────────────────────────────
        if is_main_process():
            logger.info("Iteration %d: generating %d games...", iteration, args.games_per_iter)

        # Check for shutdown before starting expensive self-play
        if _shutdown_requested:
            break

        selfplay_start = time.time()
        raw_model.eval()
        games = self_play_engine.generate_games(
            args.games_per_iter, shutdown_event=_shutdown_event,
        )
        selfplay_time = time.time() - selfplay_start

        # If shutdown was requested during self-play, skip to checkpoint
        if _shutdown_requested:
            logger.info("Self-play interrupted — skipping to checkpoint")
            break

        # Light GC between phases (skip empty_cache — costs ~200ms, unnecessary
        # on dedicated GPU; only helps when VRAM is tight from co-location)
        gc.collect()

        total_positions = sum(len(g) for g in games)
        game_lengths = [len(g) for g in games]
        avg_game_len = sum(game_lengths) / max(len(game_lengths), 1)
        outcomes = {"white": 0, "black": 0, "draw": 0}
        for g in games:
            if g.outcome > 0.5:
                outcomes["white"] += 1
            elif g.outcome < -0.5:
                outcomes["black"] += 1
            else:
                outcomes["draw"] += 1

        for game in games:
            replay_buffer.add_game(game)

        # Accumulate cumulative counters
        n_games = len(games)
        cumul_games += n_games
        cumul_positions += total_positions
        # Estimate evals from positions (each position = 1 NN eval during MCTS)
        iter_evals = total_positions * args.mcts_simulations
        cumul_evals += iter_evals

        # Win rate tracking
        win_rate = (outcomes["white"] + outcomes["black"]) / max(n_games, 1)
        recent_win_rates.append(win_rate)
        if len(recent_win_rates) > _WIN_RATE_WINDOW:
            recent_win_rates.pop(0)
        avg_win_rate = sum(recent_win_rates) / len(recent_win_rates)

        # Mean |Q| at root — diagnostic for value head flatness
        # Near zero = flat value landscape, want to see this climb
        all_root_vals = [v for g in games for v in getattr(g, "root_values", [])]
        mean_abs_q = sum(abs(v) for v in all_root_vals) / max(len(all_root_vals), 1)

        if is_main_process():
            evals_per_sec = iter_evals / max(selfplay_time, 1e-6)
            logger.info(
                "Iter %d self-play: %d games (%d total) | %d pos (%d total) | "
                "avg %.0f moves | W/B/D=%d/%d/%d (win%%=%.0f%%, trend=%.0f%%) | "
                "mean|Q|=%.3f | %.1fs (%.0f evals/s) | buffer: %d",
                iteration, n_games, cumul_games,
                total_positions, cumul_positions,
                avg_game_len,
                outcomes["white"], outcomes["black"], outcomes["draw"],
                win_rate * 100, avg_win_rate * 100,
                mean_abs_q,
                selfplay_time, evals_per_sec,
                len(replay_buffer),
            )

        # Set probe batch for monitoring (once, from first generation)
        if monitor is not None and monitor._probe_batch is None and len(replay_buffer) >= 64:
            probe_boards, *_ = replay_buffer.sample(64)
            monitor.set_probe_batch(probe_boards)

        # ── Training phase ────────────────────────────────────────
        if len(replay_buffer) < args.batch_size:
            if is_main_process():
                logger.info("Buffer too small (%d), skipping training", len(replay_buffer))
            continue

        model.train()
        train_metrics: dict[str, float] = {
            "loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "z_loss": 0.0,
            "material_loss": 0.0, "activity_loss": 0.0, "moves_left_loss": 0.0,
        }
        grad_norm_sum = 0.0
        train_start = time.time()

        # Scale training steps proportionally to buffer size so we maintain
        # roughly consistent passes through the data as the buffer grows.
        # At minimum do the base steps, scale up to 2x as buffer fills.
        if args.train_steps_scale_with_buffer:
            base_buffer = args.games_per_iter * 200  # ~1 iteration of data
            buffer_ratio = len(replay_buffer) / max(base_buffer, 1)
            num_train_steps = min(
                int(args.train_steps_per_iter * max(1.0, buffer_ratio)),
                args.train_steps_per_iter * 3,  # cap at 3x
            )
        else:
            num_train_steps = args.train_steps_per_iter

        decisive_frac_sum = 0.0
        surprise_sum = 0.0

        for step in range(num_train_steps):
            boards, target_policies, target_values, target_q_values, target_materials, target_activities, surprises, target_moves_left = (
                replay_buffer.sample(args.batch_size)
            )

            # Track PER effectiveness: fraction of batch that's decisive
            decisive_frac_sum += (target_values.abs() > 0.5).float().mean().item()

            boards = boards.to(device)
            target_policies = target_policies.to(device)
            target_values = target_values.to(device)
            target_q_values = target_q_values.to(device)
            target_materials = target_materials.to(device)
            target_activities = target_activities.to(device)
            target_moves_left = target_moves_left.to(device)

            # Policy surprise weighting: 50% uniform + 50% proportional to surprise
            surprise_mean = surprises.mean().item()
            surprise_sum += surprise_mean
            if surprise_mean > 1e-8:
                sample_weights = 0.5 + 0.5 * (surprises / max(surprise_mean, 1e-8))
            else:
                sample_weights = torch.ones_like(surprises)
            sample_weights = sample_weights.to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                policy_logits, wdl_logits, material_pred, activity_pred, moves_left_pred = model(boards)
                losses = compute_loss(
                    policy_logits, wdl_logits, material_pred, activity_pred, moves_left_pred,
                    target_policies, target_values, target_q_values,
                    target_materials, target_activities, target_moves_left,
                    z_loss_weight=config.z_loss_weight,
                    q_blend=args.q_blend,
                    sample_weights=sample_weights,
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
            grad_norm_sum += grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

        # Average metrics
        num_steps = num_train_steps
        for k in train_metrics:
            train_metrics[k] /= num_steps
        avg_grad_norm = grad_norm_sum / num_steps
        avg_decisive_frac = decisive_frac_sum / num_steps
        train_time = time.time() - train_start

        iter_time = time.time() - iter_start

        # ── Logging ───────────────────────────────────────────────
        if is_main_process():
            lrs = scheduler.get_last_lr()
            draw_rate = outcomes["draw"] / max(len(games), 1)
            logger.info(
                "Iter %d train [step %d]: loss=%.4f (pol=%.4f val=%.4f z=%.6f mat=%.4f act=%.4f mlh=%.4f) "
                "grad=%.2f decisive=%.0f%% | %.1fs (%.0fs total)",
                iteration,
                global_train_step,
                train_metrics["loss"],
                train_metrics["policy_loss"],
                train_metrics["value_loss"],
                train_metrics["z_loss"],
                train_metrics["material_loss"],
                train_metrics["activity_loss"],
                train_metrics["moves_left_loss"],
                grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                avg_decisive_frac * 100,
                train_time,
                iter_time,
            )

            if wandb_run is not None:
                log_dict = {
                    f"train/{k}": v for k, v in train_metrics.items()
                }
                log_dict.update({
                    "train/grad_norm": avg_grad_norm,
                    "train/muon_lr": lrs["muon_lr"],
                    "train/adamw_lr": lrs["adamw_lr"],
                    "train/buffer_size": len(replay_buffer),
                    "train/global_step": global_train_step,
                    # Timing
                    "time/iter_total": iter_time,
                    "time/selfplay": selfplay_time,
                    "time/training": train_time,
                    "time/selfplay_pct": selfplay_time / max(iter_time, 1e-6) * 100,
                    # Game stats (per iteration)
                    "selfplay/positions": total_positions,
                    "selfplay/games": n_games,
                    "selfplay/avg_game_length": avg_game_len,
                    "selfplay/white_wins": outcomes["white"],
                    "selfplay/black_wins": outcomes["black"],
                    "selfplay/draws": outcomes["draw"],
                    "selfplay/draw_rate": draw_rate,
                    "selfplay/win_rate": win_rate,
                    "selfplay/win_rate_trend": avg_win_rate,
                    "selfplay/positions_per_sec": total_positions / max(selfplay_time, 1e-6),
                    "selfplay/evals_per_sec": iter_evals / max(selfplay_time, 1e-6),
                    "selfplay/mean_abs_q": mean_abs_q,
                    # Cumulative totals
                    "cumulative/games": cumul_games,
                    "cumulative/positions": cumul_positions,
                    "cumulative/evals": cumul_evals,
                    # Training throughput
                    "train/steps_per_sec": num_steps / max(train_time, 1e-6),
                    "train/num_steps": num_steps,
                    # PER / signal quality
                    "train/decisive_frac": avg_decisive_frac,
                    "train/mean_surprise": surprise_sum / max(num_steps, 1),
                })
                # Don't log yet — accumulate all metrics for this iteration
                iter_log_dict = log_dict

        # ── Geometric monitoring ──────────────────────────────────
        if monitor is not None and is_main_process():
            if global_train_step % args.monitor_tier1_every == 0:
                raw_model.eval()
                probe = monitor._probe_batch
                if probe is not None:
                    with torch.no_grad():
                        p_logits, *_ = raw_model(probe.to(device))
                    geo_metrics = monitor.tier1(global_train_step, policy_logits=p_logits)
                    if wandb_run is not None:
                        iter_log_dict.update(geo_metrics)

            if global_train_step % args.monitor_tier2_every == 0:
                raw_model.eval()
                geo_metrics = monitor.tier2(global_train_step)
                if wandb_run is not None:
                    iter_log_dict.update(geo_metrics)

        # ── Evaluation ────────────────────────────────────────────
        if (
            evaluator is not None
            and args.eval_every > 0
            and (iteration + 1) % args.eval_every == 0
        ):
            if is_main_process():
                logger.info("Running Stockfish evaluation...")
                raw_model.eval()
                eval_results = evaluator.evaluate(
                    raw_model, early_stop=lambda: _shutdown_requested,
                )

                if wandb_run is not None:
                    iter_log_dict.update(eval_results)

        # ── Flush all metrics for this iteration to wandb ─────────
        if wandb_run is not None and is_main_process():
            if iter_log_dict:
                iter_log_dict["iteration"] = iteration
                wandb_run.log(iter_log_dict, commit=True)
                logger.debug("Logged %d metrics to wandb (iter %d)", len(iter_log_dict), iteration)

        # ── Checkpoint ────────────────────────────────────────────
        _should_ckpt = is_main_process() and (
            iteration == start_iteration  # always save first completed iteration
            or (args.save_every > 0 and (iteration + 1) % args.save_every == 0)
            or _shutdown_requested
        )
        if _should_ckpt:
            ckpt_path = ckpt_dir / f"iter_{iteration:06d}.pt"
            ckpt_data: dict = {
                "model": raw_model.state_dict(),
                "muon_opt": muon_opt.state_dict(),
                "adamw_opt": adamw_opt.state_dict(),
                "iteration": iteration + 1,
                "config": vars(config) if hasattr(config, '__dict__') else str(config),
            }
            if len(replay_buffer) > 0:
                ckpt_data["replay_buffer"] = replay_buffer.state_dict()
                logger.info(
                    "Including replay buffer in checkpoint (%d positions)",
                    len(replay_buffer),
                )
            torch.save(ckpt_data, ckpt_path)
            # Symlink latest
            if latest_ckpt.exists() or latest_ckpt.is_symlink():
                latest_ckpt.unlink()
            latest_ckpt.symlink_to(ckpt_path.name)
            logger.info("Saved checkpoint: %s", ckpt_path)

        if _shutdown_requested:
            logger.info("Graceful shutdown at iteration %d", iteration)
            break

    # ── Shutdown checkpoint (save latest weights if interrupted) ────────
    if _shutdown_requested and is_main_process():
        ckpt_path = ckpt_dir / f"iter_{iteration:06d}.pt"
        # Always save shutdown checkpoint (overwrite stale files from previous runs)
        logger.info("Saving shutdown checkpoint at iteration %d...", iteration)
        ckpt_data = {
            "model": raw_model.state_dict(),
            "muon_opt": muon_opt.state_dict(),
            "adamw_opt": adamw_opt.state_dict(),
            "iteration": iteration,
            "config": vars(config) if hasattr(config, '__dict__') else str(config),
        }
        if len(replay_buffer) > 0:
            ckpt_data["replay_buffer"] = replay_buffer.state_dict()
        torch.save(ckpt_data, ckpt_path)
        if latest_ckpt.exists() or latest_ckpt.is_symlink():
            latest_ckpt.unlink()
        latest_ckpt.symlink_to(ckpt_path.name)
        logger.info("Shutdown checkpoint saved: %s", ckpt_path)

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
    parser.add_argument("--attn_res_boundaries", type=str, default=None,
                        help="Explicit AttnRes boundaries, comma-separated (e.g. '0,3,7,12')")
    parser.add_argument("--no_qk_norm", action="store_true", help="Disable QK-norm")
    parser.add_argument("--activation_checkpointing", action="store_true",
                        help="Trade compute for VRAM — recompute activations during backward")

    # Training
    parser.add_argument("--num_iterations", type=int, default=1000)
    parser.add_argument("--games_per_iter", type=int, default=256)
    parser.add_argument("--train_steps_per_iter", type=int, default=1000)
    parser.add_argument("--train_steps_scale_with_buffer", action="store_true",
                        help="Scale training steps proportionally to buffer size")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--buffer_size", type=int, default=1_000_000)
    parser.add_argument("--decisive_boost", type=float, default=1.0,
                        help="PER sampling weight for decisive positions (1.0=uniform, 2.0=recommended)")
    parser.add_argument("--q_blend", type=float, default=0.0,
                        help="Blend factor for value target: 0.0=pure z (game outcome), 0.75=mostly q (MCTS root)")
    parser.add_argument("--playout_cap_fraction", type=float, default=1.0,
                        help="Fraction of moves that get full search (1.0=all, 0.25=KataGo-style PCR)")
    parser.add_argument("--material_adjudication_threshold", type=float, default=9.0,
                        help="Material diff for win/loss adjudication at max_moves (9.0=queen, 3.0=old default)")
    parser.add_argument("--fast_move_sims", type=int, default=0,
                        help="Gumbel sims for PCR fast moves (0=raw policy, 32-50=light search)")

    # Optimizer
    parser.add_argument("--muon_lr", type=float, default=0.02)
    parser.add_argument("--muon_wd", type=float, default=0.01)
    parser.add_argument("--adamw_lr", type=float, default=3e-4)
    parser.add_argument("--ns_coefficients", type=str, default="gram_ns")
    parser.add_argument("--warmup_steps", type=int, default=500)

    # Self-play
    parser.add_argument("--num_parallel_games", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=48,
                        help="CPU worker processes for MCTS (1 = single-process)")
    parser.add_argument("--mcts_simulations", type=int, default=800)
    parser.add_argument("--num_virtual_leaves", type=int, default=4,
                        help="Parallel leaves per game via virtual loss")
    parser.add_argument("--temperature_threshold", type=int, default=30)
    parser.add_argument("--dirichlet_epsilon", type=float, default=0.25,
                        help="Exploration noise strength at root (0.25=standard, higher=more diverse)")
    parser.add_argument("--mcts_algorithm", type=str, default="alphazero",
                        choices=["alphazero", "gumbel"],
                        help="MCTS algorithm: alphazero (PUCT) or gumbel (Sequential Halving)")
    parser.add_argument("--gumbel_K", type=int, default=16,
                        help="Gumbel: number of candidate actions at root")
    parser.add_argument("--gumbel_c_visit", type=float, default=50.0,
                        help="Gumbel: sigma scaling for completed Q-values")

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
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for fused kernels (adds ~30s startup)")

    # NCA Bootstrap
    parser.add_argument("--nca_bootstrap", action="store_true",
                        help="Run NCA pre-training before self-play")
    parser.add_argument("--nca_seed", type=int, default=17, help="Lucky number")
    parser.add_argument("--nca_num_rules", type=int, default=2000)
    parser.add_argument("--nca_sims_per_rule", type=int, default=8)
    parser.add_argument("--nca_max_pairs", type=int, default=2_000_000)
    parser.add_argument("--nca_min_steps", type=int, default=2000,
                        help="Min steps before saturation check")
    parser.add_argument("--nca_max_steps", type=int, default=20000,
                        help="Hard cap on NCA training steps")
    parser.add_argument("--nca_no_auto_stop", action="store_true",
                        help="Disable saturation auto-stop (train to max_steps or manual SIGTERM)")
    parser.add_argument("--nca_save_every", type=int, default=1000,
                        help="Save NCA intermediate checkpoint every N steps (for resume)")
    parser.add_argument("--nca_dataset", type=str, default=None,
                        help="Path to pre-generated NCA dataset .pt file (skip inline generation)")
    parser.add_argument("--nca_skip_reinit", action="store_true",
                        help="Don't reinit embeddings after NCA (for boundary analysis phase)")

    # Monitoring
    parser.add_argument("--monitor", action="store_true",
                        help="Enable geometric health monitoring")
    parser.add_argument("--monitor_tier1_every", type=int, default=500)
    parser.add_argument("--monitor_tier2_every", type=int, default=5000)

    # Logging
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_entity", type=str, default="aethera")
    parser.add_argument("--wandb_project", type=str, default="metis")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_api_key", type=str, default=None,
                        help="Explicit wandb API key")

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
