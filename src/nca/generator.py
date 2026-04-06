"""NCA trajectory generator adapted for 8x8 chess-board grids.

Generates Neural Cellular Automata trajectories on 8×8 grids to bootstrap
the chess transformer's attention circuits before self-play training.

Adaptations from luxia-base NCA generator:
- Grid size: 8×8 (matches chess board)
- States: 13 (matches piece type vocabulary)
- Channels: 1 (single grid = chess board layout)
- Patch size: 1 (each cell = 1 token → 64 tokens per frame)
- Frame encoding: 3 random global tokens + 64 cell states = 67 tokens
- Training objective: predict next frame's cell states (not next-token LM)

The core rule network, simulation, and complexity filtering are preserved
from the original luxia-base implementation.

Usage:
    python -m src.nca.generator --output data/nca_trajectories.pt --device cuda:0
"""

from __future__ import annotations

import argparse
import gzip
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.chess.board import NUM_PIECE_TYPES, NUM_GLOBAL_TOKENS, SEQ_LEN

logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────────


@dataclass
class ChessNCAConfig:
    """NCA config adapted for 8x8 chess-board grids."""

    grid_size: int = 8
    d_state: int = NUM_PIECE_TYPES  # 13 (matches piece vocabulary)
    n_groups: int = 1               # Single channel = chess board

    # Rule network defaults (overridden when mixed_complexity=True)
    kernel_size: int = 3
    hidden_dim: int = 32
    num_hidden_layers: int = 2

    # Dynamics
    identity_bias: float = 1.0
    temperature: float = 0.5

    # Trajectory
    num_steps: int = 128
    burn_in: int = 10

    # Complexity filtering
    filter_enabled: bool = True
    gzip_lower: float = 0.35   # Slightly wider range for smaller grid
    gzip_upper: float = 0.75
    filter_steps: int = 30

    # Mixed complexity sampling
    mixed_complexity: bool = True

    @property
    def cells_per_frame(self) -> int:
        return self.grid_size ** 2  # 64

    @property
    def tokens_per_frame(self) -> int:
        return self.cells_per_frame + NUM_GLOBAL_TOKENS  # 67


# ── Rule network (from luxia-base) ────────────────────────────────────────


class NCARule(nn.Module):
    """Neural network defining an NCA transition rule.

    Maps one-hot encoded grid state → next-state logits.
    Each random initialization produces a different automaton.
    """

    def __init__(
        self,
        d_state: int,
        n_groups: int,
        kernel_size: int,
        hidden_dim: int,
        num_hidden_layers: int,
    ) -> None:
        super().__init__()
        self.d_state = d_state
        self.n_groups = n_groups
        in_channels = d_state * n_groups
        out_channels = d_state * n_groups
        pad = kernel_size // 2

        self.perception = nn.Conv2d(
            in_channels, hidden_dim,
            kernel_size=kernel_size,
            padding=pad,
            padding_mode="circular",
        )

        layers: list[nn.Module] = []
        for _ in range(num_hidden_layers):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, 1))
            layers.append(nn.ReLU())
        self.hidden = nn.Sequential(*layers)

        self.output = nn.Conv2d(hidden_dim, out_channels, 1)

        self._random_init()

    def _random_init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.5)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=0.1)

    def forward(self, state_onehot: torch.Tensor) -> torch.Tensor:
        x = self.perception(state_onehot)
        x = F.relu(x)
        x = self.hidden(x)
        return self.output(x)


def sample_rule_config(config: ChessNCAConfig) -> dict:
    """Sample a random rule architecture from the design space."""
    if not config.mixed_complexity:
        return {
            "kernel_size": config.kernel_size,
            "hidden_dim": config.hidden_dim,
            "num_hidden_layers": config.num_hidden_layers,
            "identity_bias": config.identity_bias,
            "temperature": config.temperature,
        }

    return {
        "kernel_size": random.choice([3, 5]),
        "hidden_dim": random.choice([16, 32, 48]),
        "num_hidden_layers": random.choice([1, 2, 3]),
        "identity_bias": random.uniform(0.5, 2.0),
        "temperature": random.uniform(0.3, 1.0),
    }


# ── Simulation ─────────────────────────────────────────────────────────────


@torch.no_grad()
def simulate_trajectory(
    rule: NCARule,
    config: ChessNCAConfig,
    identity_bias: float,
    temperature: float,
    batch_size: int = 1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Simulate an NCA trajectory on an 8x8 grid.

    Returns:
        states: (batch, num_steps, n_groups, 8, 8) integer cell states [0, d_state)
    """
    H = W = config.grid_size
    d = config.d_state
    G = config.n_groups

    state = torch.randint(0, d, (batch_size, G, H, W), device=device)

    def step(s: torch.Tensor) -> torch.Tensor:
        s_flat = s.reshape(batch_size * G, H, W)
        onehot = F.one_hot(s_flat.long(), d).permute(0, 3, 1, 2).float()
        onehot = onehot.reshape(batch_size, G * d, H, W)

        logits = rule(onehot)
        if identity_bias > 0:
            logits = logits + identity_bias * onehot

        logits = logits.reshape(batch_size, G, d, H, W)
        logits = logits.permute(0, 1, 3, 4, 2).reshape(-1, d)

        probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
        next_state = torch.multinomial(probs, 1)
        return next_state.reshape(batch_size, G, H, W)

    # Burn-in
    for _ in range(config.burn_in):
        state = step(state)

    # Record
    trajectory = torch.zeros(
        batch_size, config.num_steps, G, H, W, dtype=torch.long, device=device,
    )
    for t in range(config.num_steps):
        trajectory[:, t] = state
        state = step(state)

    return trajectory


# ── Frame encoding (matching chess board format) ───────────────────────────


def encode_frame(
    grid: torch.Tensor,
) -> torch.Tensor:
    """Encode an 8x8 grid state as a 67-token sequence matching chess format.

    Args:
        grid: (8, 8) or (1, 8, 8) integer cell states [0, 12].

    Returns:
        tokens: (67,) long tensor — 3 random global tokens + 64 cell states.
    """
    if grid.dim() == 3:
        grid = grid.squeeze(0)  # Remove channel dim

    tokens = torch.zeros(SEQ_LEN, dtype=torch.long)

    # Random global tokens (castling, en_passant, side_to_move)
    # These are random noise during NCA — the model learns to ignore them
    tokens[0] = torch.randint(0, 16, (1,)).item()  # castling (0-15)
    tokens[1] = torch.randint(0, 9, (1,)).item()    # en passant (0-8)
    tokens[2] = torch.randint(0, 2, (1,)).item()    # side to move (0-1)

    # Cell states as square tokens
    tokens[3:] = grid.flatten()

    return tokens


def trajectory_to_training_pairs(
    trajectory: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert NCA trajectory to (input_frame, target_cells) pairs.

    Given a trajectory of T frames, produces T-1 training pairs where
    input is frame_t (67 tokens) and target is frame_{t+1}'s 64 cell states.

    Args:
        trajectory: (batch, T, n_groups, 8, 8) integer states.

    Returns:
        inputs: (N, 67) long tensor — encoded input frames.
        targets: (N, 64) long tensor — next frame cell states (0-12).
    """
    B, T, G, H, W = trajectory.shape
    inputs_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []

    for b in range(B):
        for t in range(T - 1):
            frame_input = encode_frame(trajectory[b, t])
            frame_target = trajectory[b, t + 1].reshape(-1)  # (G*H*W,) → (64,)
            inputs_list.append(frame_input)
            targets_list.append(frame_target)

    return torch.stack(inputs_list), torch.stack(targets_list)


# ── Complexity filtering ──────────────────────────────────────────────────


def compute_gzip_complexity(trajectory: torch.Tensor, d_state: int) -> float:
    """Gzip compression ratio of a trajectory as complexity proxy."""
    raw = trajectory.cpu().numpy().astype(np.uint8).tobytes()
    compressed = gzip.compress(raw)
    return len(compressed) / max(len(raw), 1)


def evaluate_rule_complexity(
    rule: NCARule,
    config: ChessNCAConfig,
    rule_params: dict,
    device: torch.device,
    eval_sims: int = 4,
) -> float:
    """Simulate a short trajectory and return gzip complexity."""
    traj = simulate_trajectory(
        rule=rule, config=config,
        identity_bias=rule_params["identity_bias"],
        temperature=rule_params["temperature"],
        batch_size=eval_sims, device=device,
    )
    return compute_gzip_complexity(traj, config.d_state)


def generate_and_filter_rules(
    config: ChessNCAConfig,
    num_rules: int,
    device: torch.device,
) -> list[tuple[NCARule, dict]]:
    """Generate NCA rules with complexity filtering."""
    accepted: list[tuple[NCARule, dict]] = []
    candidates_tested = 0
    t0 = time.time()

    while len(accepted) < num_rules:
        params = sample_rule_config(config)
        rule = NCARule(
            d_state=config.d_state,
            n_groups=config.n_groups,
            kernel_size=params["kernel_size"],
            hidden_dim=params["hidden_dim"],
            num_hidden_layers=params["num_hidden_layers"],
        ).to(device)
        candidates_tested += 1

        if not config.filter_enabled:
            accepted.append((rule, params))
            continue

        ratio = evaluate_rule_complexity(rule, config, params, device)

        if config.gzip_lower <= ratio <= config.gzip_upper:
            accepted.append((rule, params))
            if len(accepted) % 50 == 0:
                elapsed = time.time() - t0
                rate = len(accepted) / candidates_tested
                logger.info(
                    "  %d/%d rules (%.0f%% accept, %.1fs, gzip=%.3f)",
                    len(accepted), num_rules, rate * 100, elapsed, ratio,
                )

    elapsed = time.time() - t0
    rate = num_rules / max(candidates_tested, 1)
    logger.info(
        "Rules: %d from %d candidates (%.0f%%, %.1fs)",
        num_rules, candidates_tested, rate * 100, elapsed,
    )
    return accepted


# ── Dataset generation ────────────────────────────────────────────────────


def generate_nca_dataset(
    config: ChessNCAConfig,
    num_rules: int = 2000,
    sims_per_rule: int = 8,
    device: torch.device = torch.device("cpu"),
    max_pairs: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate NCA training dataset as (input, target) frame pairs.

    Args:
        config: NCA configuration.
        num_rules: Number of unique NCA rules to generate.
        sims_per_rule: Trajectories per rule.
        device: GPU for simulation.
        max_pairs: Cap on total training pairs (None = unlimited).

    Returns:
        inputs: (N, 67) long tensor — encoded input frames.
        targets: (N, 64) long tensor — next-frame cell states.
    """
    logger.info("=" * 60)
    logger.info("NCA Dataset Generation (8x8 chess-board grids)")
    logger.info("=" * 60)
    logger.info("Grid: %dx%d, %d states, %d channels", 8, 8, config.d_state, config.n_groups)
    logger.info("Steps: %d (burn-in: %d)", config.num_steps, config.burn_in)
    logger.info("Rules: %d, sims/rule: %d", num_rules, sims_per_rule)

    pairs_per_rule = sims_per_rule * (config.num_steps - 1)
    estimated_total = num_rules * pairs_per_rule
    if max_pairs is not None:
        estimated_total = min(estimated_total, max_pairs)
    logger.info("Estimated pairs: %d (%.1fM tokens)", estimated_total, estimated_total * 67 / 1e6)

    # Generate rules
    rules = generate_and_filter_rules(config, num_rules, device)

    # Generate trajectories and create training pairs
    all_inputs: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    total_pairs = 0
    t0 = time.time()

    for i, (rule, params) in enumerate(rules):
        traj = simulate_trajectory(
            rule=rule, config=config,
            identity_bias=params["identity_bias"],
            temperature=params["temperature"],
            batch_size=sims_per_rule, device=device,
        )

        inputs, targets = trajectory_to_training_pairs(traj.cpu())
        all_inputs.append(inputs)
        all_targets.append(targets)
        total_pairs += len(inputs)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            logger.info(
                "  %d/%d rules, %d pairs, %.0f pairs/s",
                i + 1, len(rules), total_pairs, total_pairs / max(elapsed, 1e-6),
            )

        if max_pairs is not None and total_pairs >= max_pairs:
            break

    inputs = torch.cat(all_inputs, dim=0)
    targets = torch.cat(all_targets, dim=0)

    if max_pairs is not None and len(inputs) > max_pairs:
        inputs = inputs[:max_pairs]
        targets = targets[:max_pairs]

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(
        "Done: %d pairs from %d rules, %.1fs (%.0f pairs/s)",
        len(inputs), min(len(rules), i + 1 if 'i' in dir() else len(rules)),
        elapsed, len(inputs) / max(elapsed, 1e-6),
    )

    return inputs, targets


# ── CLI ───────────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Generate 8x8 NCA dataset for chess")
    p.add_argument("--output", type=str, required=True, help="Output .pt file path")
    p.add_argument("--num_rules", type=int, default=2000)
    p.add_argument("--sims_per_rule", type=int, default=8)
    p.add_argument("--max_pairs", type=int, default=None)
    p.add_argument("--num_steps", type=int, default=128)
    p.add_argument("--burn_in", type=int, default=10)
    p.add_argument("--gzip_lower", type=float, default=0.35)
    p.add_argument("--gzip_upper", type=float, default=0.75)
    p.add_argument("--no_filter", action="store_true")
    p.add_argument("--no_mixed", action="store_true")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = ChessNCAConfig(
        num_steps=args.num_steps,
        burn_in=args.burn_in,
        filter_enabled=not args.no_filter,
        gzip_lower=args.gzip_lower,
        gzip_upper=args.gzip_upper,
        mixed_complexity=not args.no_mixed,
    )

    inputs, targets = generate_nca_dataset(
        config=config,
        num_rules=args.num_rules,
        sims_per_rule=args.sims_per_rule,
        device=torch.device(args.device),
        max_pairs=args.max_pairs,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"inputs": inputs, "targets": targets}, output_path)
    logger.info("Saved to %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
