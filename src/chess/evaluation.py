"""Stockfish evaluation gauntlet.

Plays the model against Stockfish at various depths to measure Elo strength.
Uses python-chess's UCI engine interface.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import chess
import chess.engine
import torch

from src.chess.board import BoardEncoder, MoveEncoder
from src.chess.mcts import MCTS, MCTSConfig

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    stockfish_path: str = "stockfish"
    depths: list[int] = field(default_factory=lambda: [1, 3, 5, 8])
    games_per_depth: int = 50
    mcts_simulations: int = 400  # Fewer sims for eval speed
    cpuct: float = 1.25
    max_moves: int = 300


def _elo_diff(wins: int, draws: int, losses: int) -> float:
    """Calculate Elo difference from match result.

    Uses: elo_diff = -400 * log10(1/score - 1)
    where score = (wins + 0.5 * draws) / total_games
    """
    total = wins + draws + losses
    if total == 0:
        return 0.0
    score = (wins + 0.5 * draws) / total
    # Clamp to avoid log(0) or log(negative)
    score = max(0.001, min(0.999, score))
    return -400.0 * math.log10(1.0 / score - 1.0)


class StockfishEvaluator:
    """Evaluates model strength by playing against Stockfish."""

    def __init__(self, config: EvalConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self._engine: Optional[chess.engine.SimpleEngine] = None

    def _get_engine(self) -> chess.engine.SimpleEngine:
        if self._engine is None:
            try:
                self._engine = chess.engine.SimpleEngine.popen_uci(
                    self.config.stockfish_path,
                )
            except FileNotFoundError:
                raise RuntimeError(
                    f"Stockfish not found at '{self.config.stockfish_path}'. "
                    f"Install stockfish or set the correct path."
                )
        return self._engine

    def close(self) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def evaluate(self, model: torch.nn.Module) -> dict[str, float]:
        """Play a gauntlet against Stockfish at various depths.

        Args:
            model: Chess transformer model.

        Returns:
            Dict of metrics suitable for wandb logging.
        """
        mcts_config = MCTSConfig(
            num_simulations=self.config.mcts_simulations,
            cpuct=self.config.cpuct,
            dirichlet_alpha=0.0,   # No noise during evaluation
            dirichlet_epsilon=0.0,
        )
        mcts = MCTS(model=model, config=mcts_config, device=self.device)

        results: dict[str, float] = {}
        engine = self._get_engine()

        for depth in self.config.depths:
            wins, draws, losses = 0, 0, 0

            for game_idx in range(self.config.games_per_depth):
                model_is_white = game_idx % 2 == 0
                outcome = self._play_game(mcts, engine, depth, model_is_white)

                if outcome == 1:
                    wins += 1
                elif outcome == 0:
                    draws += 1
                else:
                    losses += 1

            elo = _elo_diff(wins, draws, losses)
            total = wins + draws + losses
            prefix = f"eval/sf_d{depth}"
            results[f"{prefix}/wins"] = wins
            results[f"{prefix}/draws"] = draws
            results[f"{prefix}/losses"] = losses
            results[f"{prefix}/win_rate"] = (wins + 0.5 * draws) / total if total > 0 else 0.0
            results[f"{prefix}/elo_diff"] = elo

            logger.info(
                "vs Stockfish depth %d: +%d =%d -%d (Elo diff: %+.0f)",
                depth, wins, draws, losses, elo,
            )

        return results

    def _play_game(
        self,
        mcts: MCTS,
        engine: chess.engine.SimpleEngine,
        sf_depth: int,
        model_is_white: bool,
    ) -> int:
        """Play one game. Returns +1 (model wins), 0 (draw), -1 (model loses)."""
        board = chess.Board()
        move_count = 0

        while not board.is_game_over(claim_draw=True) and move_count < self.config.max_moves:
            is_model_turn = (board.turn == chess.WHITE) == model_is_white

            if is_model_turn:
                # Model's turn: use MCTS
                visit_counts = mcts.search(board)
                if not visit_counts:
                    break
                move = mcts.select_move(visit_counts, temperature=0)
            else:
                # Stockfish's turn
                result = engine.play(board, chess.engine.Limit(depth=sf_depth))
                if result.move is None:
                    break
                move = result.move

            board.push(move)
            move_count += 1

        # Determine outcome from model's perspective
        result = board.result(claim_draw=True)
        if result == "1-0":
            return 1 if model_is_white else -1
        elif result == "0-1":
            return -1 if model_is_white else 1
        return 0
