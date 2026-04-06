"""Self-play game generation for AlphaZero-style training.

Generates games by playing the current model against itself using MCTS.
Each game produces training examples: (board_encoding, mcts_policy, game_outcome).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import chess
import torch

from src.chess.board import BoardEncoder, MoveEncoder, SEQ_LEN, POLICY_SIZE
from src.chess.mcts import MCTS, BatchedMCTS, MCTSConfig

logger = logging.getLogger(__name__)


@dataclass
class GameRecord:
    """One completed self-play game."""

    positions: list[torch.Tensor] = field(default_factory=list)   # Each (67,) long
    policies: list[torch.Tensor] = field(default_factory=list)    # Each (4672,) float
    outcome: float = 0.0  # +1 white wins, -1 black wins, 0 draw

    def __len__(self) -> int:
        return len(self.positions)


@dataclass
class SelfPlayConfig:
    """Self-play generation configuration."""

    num_parallel: int = 64
    mcts_simulations: int = 800
    cpuct: float = 1.25
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temperature_threshold: int = 30  # Ply after which to play greedily
    max_moves: int = 512             # Draw if exceeded

    def to_mcts_config(self) -> MCTSConfig:
        return MCTSConfig(
            num_simulations=self.mcts_simulations,
            cpuct=self.cpuct,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon,
        )


class SelfPlayWorker:
    """Generates self-play games using batched MCTS."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: SelfPlayConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.mcts = BatchedMCTS(
            model=model,
            config=config.to_mcts_config(),
            device=device,
            num_parallel=config.num_parallel,
        )
        # For single-game MCTS
        self._single_mcts = MCTS(
            model=model,
            config=config.to_mcts_config(),
            device=device,
        )

    def generate_games(self, num_games: int) -> list[GameRecord]:
        """Generate self-play games.

        Runs games in batches of `num_parallel`. Each game continues until
        checkmate, stalemate, draw, or max_moves is reached.

        Args:
            num_games: Total number of games to generate.

        Returns:
            List of completed GameRecord objects.
        """
        completed: list[GameRecord] = []
        remaining = num_games

        while remaining > 0:
            batch_size = min(remaining, self.config.num_parallel)
            batch_games = self._play_batch(batch_size)
            completed.extend(batch_games)
            remaining -= batch_size
            logger.info(
                "Generated %d/%d games (batch of %d)",
                len(completed), num_games, batch_size,
            )

        return completed

    def _play_batch(self, batch_size: int) -> list[GameRecord]:
        """Play a batch of games in parallel."""
        boards = [chess.Board() for _ in range(batch_size)]
        records = [GameRecord() for _ in range(batch_size)]
        active = list(range(batch_size))  # Indices of games still in progress
        ply_counts = [0] * batch_size

        while active:
            # Get active boards
            active_boards = [boards[i] for i in active]

            # Run batched MCTS search
            visit_counts_list = self.mcts.search_batch(active_boards)

            # Process results for each active game
            next_active: list[int] = []
            for j, game_idx in enumerate(active):
                visit_counts = visit_counts_list[j]
                board = boards[game_idx]
                record = records[game_idx]

                if not visit_counts:
                    # Game is over (no legal moves / draw detected)
                    record.outcome = self._get_outcome(board)
                    continue

                # Record position and MCTS policy
                record.positions.append(BoardEncoder.encode_board(board))
                record.policies.append(MoveEncoder.encode_policy(visit_counts))

                # Select move with temperature scheduling
                ply = ply_counts[game_idx]
                temp = self.config.temperature if ply < self.config.temperature_threshold else 0.0
                move = self._single_mcts.select_move(visit_counts, temperature=temp)

                # Play the move
                board.push(move)
                ply_counts[game_idx] += 1

                # Check termination
                if board.is_game_over(claim_draw=True):
                    record.outcome = self._get_outcome(board)
                elif ply_counts[game_idx] >= self.config.max_moves:
                    record.outcome = 0.0  # Draw by max moves
                else:
                    next_active.append(game_idx)

            active = next_active

        return records

    @staticmethod
    def _get_outcome(board: chess.Board) -> float:
        """Get game outcome: +1 white wins, -1 black wins, 0 draw."""
        result = board.result(claim_draw=True)
        if result == "1-0":
            return 1.0
        elif result == "0-1":
            return -1.0
        return 0.0
