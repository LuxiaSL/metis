"""Stockfish evaluation gauntlet.

Plays the model against Stockfish at various depths to measure Elo strength.
Uses BatchedMCTS for parallel game evaluation (much faster than sequential).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import chess
import chess.engine
import torch

from src.chess.board import BoardEncoder, MoveEncoder
from src.chess.mcts import BatchedMCTS, MCTSConfig, select_move

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    stockfish_path: str = "stockfish"
    depths: list[int] = field(default_factory=lambda: [1, 3, 5])
    games_per_depth: int = 20
    mcts_simulations: int = 200
    cpuct: float = 1.25
    max_moves: int = 300
    num_parallel: int = 10  # Games to run simultaneously


def _elo_diff(wins: int, draws: int, losses: int) -> float:
    """Calculate Elo difference from match result."""
    total = wins + draws + losses
    if total == 0:
        return 0.0
    score = (wins + 0.5 * draws) / total
    score = max(0.001, min(0.999, score))
    return -400.0 * math.log10(1.0 / score - 1.0)


@dataclass
class _GameState:
    """State for one eval game."""

    board: chess.Board
    model_is_white: bool
    move_count: int = 0
    done: bool = False
    outcome: int = 0  # +1 model win, 0 draw, -1 model loss


class StockfishEvaluator:
    """Evaluates model strength by playing against Stockfish.

    Uses BatchedMCTS to evaluate multiple games in parallel, batching
    the model's NN inference across all active games for much better
    GPU utilization than sequential single-game evaluation.
    """

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

    def evaluate(
        self,
        model: torch.nn.Module,
        early_stop: Optional[callable] = None,
    ) -> dict[str, float]:
        """Play a gauntlet against Stockfish at various depths.

        Runs multiple games in parallel using BatchedMCTS for efficient
        GPU utilization. Falls back gracefully on early_stop.

        Args:
            model: Chess transformer model.
            early_stop: Optional callable returning True to interrupt eval.

        Returns:
            Dict of metrics suitable for wandb logging.
        """
        mcts_config = MCTSConfig(
            num_simulations=self.config.mcts_simulations,
            cpuct=self.config.cpuct,
            dirichlet_alpha=0.0,
            dirichlet_epsilon=0.0,
        )
        batched_mcts = BatchedMCTS(
            model=model, config=mcts_config, device=self.device,
        )

        results: dict[str, float] = {}
        engine = self._get_engine()
        _interrupted = False

        for depth in self.config.depths:
            if _interrupted:
                break

            wins, draws, losses = 0, 0, 0
            games_started = 0
            eval_start = time.time()
            logger.info(
                "Starting eval vs Stockfish depth %d (%d games, %d parallel)...",
                depth, self.config.games_per_depth, self.config.num_parallel,
            )

            # Process games in parallel batches
            while games_started < self.config.games_per_depth:
                if early_stop is not None and early_stop():
                    logger.info("Eval interrupted by shutdown request")
                    _interrupted = True
                    break

                # Start a batch of games
                batch_size = min(
                    self.config.num_parallel,
                    self.config.games_per_depth - games_started,
                )
                games = [
                    _GameState(
                        board=chess.Board(),
                        model_is_white=(games_started + i) % 2 == 0,
                    )
                    for i in range(batch_size)
                ]
                games_started += batch_size

                # Play batch to completion
                self._play_batch(games, batched_mcts, engine, depth, early_stop)

                # Tally results
                for g in games:
                    if g.outcome == 1:
                        wins += 1
                    elif g.outcome == -1:
                        losses += 1
                    else:
                        draws += 1

                completed = wins + draws + losses
                logger.info(
                    "  Stockfish d%d: %d/%d games (+%d =%d -%d)",
                    depth, completed, self.config.games_per_depth,
                    wins, draws, losses,
                )

            total = wins + draws + losses
            if total > 0:
                elo = _elo_diff(wins, draws, losses)
                elapsed = time.time() - eval_start
                prefix = f"eval/sf_d{depth}"
                results[f"{prefix}/wins"] = wins
                results[f"{prefix}/draws"] = draws
                results[f"{prefix}/losses"] = losses
                results[f"{prefix}/win_rate"] = (wins + 0.5 * draws) / total
                results[f"{prefix}/elo_diff"] = elo

                logger.info(
                    "vs Stockfish depth %d: +%d =%d -%d (Elo diff: %+.0f) in %.0fs%s",
                    depth, wins, draws, losses, elo, elapsed,
                    " [partial]" if _interrupted else "",
                )

        return results

    def _play_batch(
        self,
        games: list[_GameState],
        batched_mcts: BatchedMCTS,
        engine: chess.engine.SimpleEngine,
        sf_depth: int,
        early_stop: Optional[callable] = None,
    ) -> None:
        """Play a batch of games to completion with batched model inference."""
        max_moves = self.config.max_moves

        while any(not g.done for g in games):
            if early_stop is not None and early_stop():
                for g in games:
                    g.done = True
                break

            # ── Model turns: batch MCTS across all model-turn games ──
            model_indices = [
                i for i, g in enumerate(games)
                if not g.done
                and not g.board.is_game_over(claim_draw=True)
                and (g.board.turn == chess.WHITE) == g.model_is_white
            ]

            if model_indices:
                boards = [games[i].board for i in model_indices]
                all_vc = batched_mcts.search_batch(boards)
                for j, idx in enumerate(model_indices):
                    vc = all_vc[j]
                    if vc:
                        move = select_move(vc, temperature=0)
                        games[idx].board.push(move)
                        games[idx].move_count += 1

            # ── Stockfish turns: sequential (fast at low depth) ──
            sf_indices = [
                i for i, g in enumerate(games)
                if not g.done
                and not g.board.is_game_over(claim_draw=True)
                and (g.board.turn == chess.WHITE) != g.model_is_white
            ]

            for idx in sf_indices:
                try:
                    result = engine.play(
                        games[idx].board, chess.engine.Limit(depth=sf_depth),
                    )
                    if result.move is not None:
                        games[idx].board.push(result.move)
                        games[idx].move_count += 1
                except chess.engine.EngineTerminatedError:
                    logger.warning("Stockfish engine terminated, restarting")
                    self._engine = None
                    engine = self._get_engine()
                    games[idx].done = True

            # ── Check for game completion ──
            for g in games:
                if g.done:
                    continue
                if g.board.is_game_over(claim_draw=True) or g.move_count >= max_moves:
                    g.done = True
                    result = g.board.result(claim_draw=True)
                    if result == "1-0":
                        g.outcome = 1 if g.model_is_white else -1
                    elif result == "0-1":
                        g.outcome = -1 if g.model_is_white else 1
                    else:
                        g.outcome = 0
