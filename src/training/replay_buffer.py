"""Circular experience replay buffer for self-play training data.

Stores (board_encoding, policy_target, value_target) tuples from self-play games.
Supports uniform random sampling for training batches.
"""

from __future__ import annotations

import torch

from src.chess.self_play import GameRecord
from src.chess.board import SEQ_LEN, POLICY_SIZE


class ReplayBuffer:
    """Fixed-capacity circular buffer for self-play positions."""

    def __init__(self, capacity: int = 1_000_000) -> None:
        self.capacity = capacity
        self.boards = torch.zeros(capacity, SEQ_LEN, dtype=torch.long)
        self.policies = torch.zeros(capacity, POLICY_SIZE, dtype=torch.float32)
        self.values = torch.zeros(capacity, dtype=torch.float32)
        self._index: int = 0
        self._size: int = 0

    def __len__(self) -> int:
        return self._size

    def add_game(self, record: GameRecord) -> None:
        """Add all positions from a completed game.

        Values are stored from the perspective of the side to move at each
        position. White positions get outcome directly; black positions get
        -outcome (since outcome is from white's perspective).
        """
        for i, (board, policy) in enumerate(zip(record.positions, record.policies)):
            # Even ply = white to move, odd ply = black to move
            # outcome is from white's perspective (+1 = white wins)
            value = record.outcome if i % 2 == 0 else -record.outcome

            self.boards[self._index] = board
            self.policies[self._index] = policy
            self.values[self._index] = value

            self._index = (self._index + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random batch of training examples.

        Returns:
            boards: (B, 67) long tensor
            policies: (B, 4672) float tensor (probability distributions)
            values: (B,) float tensor (game outcomes from side-to-move perspective)

        Raises:
            ValueError: If buffer has fewer samples than batch_size.
        """
        if self._size < batch_size:
            raise ValueError(
                f"Buffer has {self._size} samples, need {batch_size}"
            )
        indices = torch.randint(0, self._size, (batch_size,))
        return (
            self.boards[indices],
            self.policies[indices],
            self.values[indices],
        )

    def clear(self) -> None:
        """Reset the buffer."""
        self._index = 0
        self._size = 0
