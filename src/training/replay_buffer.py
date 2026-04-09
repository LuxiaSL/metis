"""Circular experience replay buffer for self-play training data.

Stores (board_encoding, policy_target, value_target, q_value_target,
material_target, activity_target, surprise) tuples from self-play games.
Supports weighted sampling (PER decisive_boost) for training batches.
"""

from __future__ import annotations

import torch

from src.chess.self_play import GameRecord
from src.chess.board import SEQ_LEN, POLICY_SIZE

# Material value per piece token index (0=empty, 1-6=white P/N/B/R/Q/K, 7-12=black)
_TOKEN_MATERIAL = torch.tensor(
    [0.0, 1.0, 3.0, 3.0, 5.0, 9.0, 0.0, -1.0, -3.0, -3.0, -5.0, -9.0, 0.0],
    dtype=torch.float32,
)
_MATERIAL_NORM = 39.0  # max per-side material (standard starting)
_ACTIVITY_NORM = 40.0  # typical legal move count
_MOVES_LEFT_NORM = 150.0  # typical max game length in plies for normalization


class ReplayBuffer:
    """Fixed-capacity circular buffer for self-play positions."""

    def __init__(self, capacity: int = 1_000_000, decisive_boost: float = 1.0) -> None:
        self.capacity = capacity
        self.decisive_boost = decisive_boost
        self.boards = torch.zeros(capacity, SEQ_LEN, dtype=torch.long)
        self.policies = torch.zeros(capacity, POLICY_SIZE, dtype=torch.float32)
        self.values = torch.zeros(capacity, dtype=torch.float32)
        self.q_wdl = torch.zeros(capacity, 3, dtype=torch.float32)  # Full WDL probs from MCTS root
        self.materials = torch.zeros(capacity, dtype=torch.float32)
        self.activities = torch.zeros(capacity, dtype=torch.float32)
        self.surprises = torch.ones(capacity, dtype=torch.float32)  # KL(improved || prior), default 1.0
        self.moves_left = torch.zeros(capacity, dtype=torch.float32)  # normalized remaining moves
        self.weights = torch.ones(capacity, dtype=torch.float32)
        self._index: int = 0
        self._size: int = 0

    def __len__(self) -> int:
        return self._size

    @staticmethod
    def _compute_material(board_tokens: torch.Tensor) -> float:
        """Compute normalized material balance from board token encoding.

        Args:
            board_tokens: (67,) long tensor — 3 global tokens + 64 square tokens.

        Returns:
            Normalized material balance (positive = white advantage), roughly in [-1, 1].
        """
        pieces = board_tokens[3:67]  # 64 square tokens
        material = _TOKEN_MATERIAL[pieces].sum().item()
        return material / _MATERIAL_NORM

    def add_game(self, record: GameRecord) -> None:
        """Add all positions from a completed game.

        Values (z-targets) are stored from the perspective of the side to move.
        White positions get outcome directly; black positions get -outcome.
        Q-values (MCTS root values) are also stored per position for blended
        value targets. With PCR, only training moves have positions recorded,
        so the ply index tracks position order (not absolute game ply).

        Material is computed from board tokens (deterministic, no chess.Board needed).
        Activity is taken from GameRecord if available, else defaults to 0.
        """
        has_activities = hasattr(record, "activities") and len(record.activities) > 0
        has_root_wdl = hasattr(record, "root_wdl") and len(record.root_wdl) > 0
        has_surprise = hasattr(record, "surprise") and len(record.surprise) > 0
        has_plies = hasattr(record, "plies") and len(record.plies) > 0
        total_plies = getattr(record, "total_plies", 0)

        for i, (board, policy) in enumerate(zip(record.positions, record.policies)):
            # Even ply = white to move, odd ply = black to move
            # outcome is from white's perspective (+1 = white wins)
            value = record.outcome if i % 2 == 0 else -record.outcome

            self.boards[self._index] = board
            self.policies[self._index] = policy
            self.values[self._index] = value

            # Q-WDL: full WDL distribution from MCTS root
            if has_root_wdl and i < len(record.root_wdl):
                self.q_wdl[self._index] = torch.tensor(record.root_wdl[i], dtype=torch.float32)
            else:
                # Fallback: convert z-target to hard WDL
                self.q_wdl[self._index] = torch.tensor(
                    [max(0.0, -value), 1.0 - abs(value), max(0.0, value)],
                    dtype=torch.float32,
                )

            self.materials[self._index] = self._compute_material(board)

            if has_activities and i < len(record.activities):
                self.activities[self._index] = record.activities[i]

            if has_surprise and i < len(record.surprise):
                self.surprises[self._index] = record.surprise[i]
            else:
                self.surprises[self._index] = 1.0  # Default: no surprise weighting

            # Moves-left: (total_plies - ply_at_position) / norm, clamped to [0, 1]
            if has_plies and i < len(record.plies) and total_plies > 0:
                raw = max(0, total_plies - record.plies[i]) / _MOVES_LEFT_NORM
                self.moves_left[self._index] = min(raw, 1.0)
            else:
                self.moves_left[self._index] = 0.0

            # PER: decisive positions (win/loss) get higher sampling weight
            weight = self.decisive_boost if abs(record.outcome) > 0.5 else 1.0
            self.weights[self._index] = weight

            self._index = (self._index + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random batch of training examples.

        Returns:
            boards: (B, 67) long tensor
            policies: (B, 4672) float tensor (probability distributions)
            values: (B,) float tensor — z-targets (game outcomes, side-to-move perspective)
            q_wdl: (B, 3) float tensor — q-targets (full WDL probs from MCTS root)
            materials: (B,) float tensor (normalized material balance)
            activities: (B,) float tensor (normalized legal move count)
            surprises: (B,) float tensor — KL(improved || prior) per position
            moves_left: (B,) float tensor — normalized remaining game length

        Raises:
            ValueError: If buffer has fewer samples than batch_size.
        """
        if self._size < batch_size:
            raise ValueError(
                f"Buffer has {self._size} samples, need {batch_size}"
            )
        # Weighted sampling: decisive_boost > 1.0 upweights win/loss positions
        if self.decisive_boost > 1.0:
            probs = self.weights[:self._size]
            probs = probs / probs.sum()
            indices = torch.multinomial(probs, batch_size, replacement=True)
        else:
            indices = torch.randint(0, self._size, (batch_size,))
        return (
            self.boards[indices],
            self.policies[indices],
            self.values[indices],
            self.q_wdl[indices],
            self.materials[indices],
            self.activities[indices],
            self.surprises[indices],
            self.moves_left[indices],
        )

    def clear(self) -> None:
        """Reset the buffer."""
        self._index = 0
        self._size = 0

    def state_dict(self) -> dict:
        """Serialize buffer state for checkpointing.

        Only saves the populated portion to reduce checkpoint size.
        Policies are stored as float16 (~2x smaller, precision is fine for targets).
        """
        n = self._size
        return {
            "boards": self.boards[:n].clone(),
            "policies": self.policies[:n].to(torch.float16).clone(),
            "values": self.values[:n].clone(),
            "q_wdl": self.q_wdl[:n].clone(),
            "materials": self.materials[:n].clone(),
            "activities": self.activities[:n].clone(),
            "surprises": self.surprises[:n].clone(),
            "moves_left": self.moves_left[:n].clone(),
            "weights": self.weights[:n].clone(),
            "index": self._index,
            "size": self._size,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore buffer state from checkpoint.

        Handles loading from older checkpoints that may not have materials/activities.
        """
        n = state["size"]
        self.boards[:n] = state["boards"]
        self.policies[:n] = state["policies"].to(torch.float32)
        self.values[:n] = state["values"]
        self._index = state["index"]
        self._size = n

        # Backwards-compatible: older checkpoints won't have these fields
        if "materials" in state:
            self.materials[:n] = state["materials"]
        else:
            # Recompute material from board tokens
            for i in range(n):
                self.materials[i] = self._compute_material(self.boards[i])

        if "q_wdl" in state:
            self.q_wdl[:n] = state["q_wdl"]
        elif "q_values" in state:
            # Backward compat: older checkpoints had scalar q_values — convert to WDL
            qv = state["q_values"]
            self.q_wdl[:n, 0] = (-qv).clamp(min=0.0)   # p_loss
            self.q_wdl[:n, 1] = (1.0 - qv.abs())        # p_draw
            self.q_wdl[:n, 2] = qv.clamp(min=0.0)        # p_win
        else:
            # No q data at all — use z-target converted to hard WDL
            zv = self.values[:n]
            self.q_wdl[:n, 0] = (-zv).clamp(min=0.0)
            self.q_wdl[:n, 1] = (1.0 - zv.abs())
            self.q_wdl[:n, 2] = zv.clamp(min=0.0)

        if "activities" in state:
            self.activities[:n] = state["activities"]
        # else: activities stay at 0 (no way to recompute without chess.Board)

        if "surprises" in state:
            self.surprises[:n] = state["surprises"]
        # else: surprises stay at 1.0 (uniform — backward compatible)

        if "moves_left" in state:
            self.moves_left[:n] = state["moves_left"]
        # else: moves_left stay at 0.0 (no way to recompute retroactively)

        if "weights" in state:
            self.weights[:n] = state["weights"]
        # else: weights stay at 1.0 (uniform — backward compatible)
