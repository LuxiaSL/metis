"""Stockfish-anchored training positions for value/policy grounding.

Generates positions evaluated by Stockfish to anchor the value head to reality
and provide direct policy supervision on gross blunders. Breaks the self-play
circular signal where both sides converge on bad moves that go unpunished.

Two position sources:
  - Random legal positions (broad coverage)
  - Early self-play positions (targeted correction of actual openings)
"""
from __future__ import annotations

import logging
import math
import random
from typing import Optional

import chess
import chess.engine
import numpy as np
import torch

from src.chess.board import BoardEncoder, MoveEncoder, POLICY_SIZE, mirror_policy
from src.chess.self_play import GameRecord

logger = logging.getLogger(__name__)

# Logistic curve parameters for centipawn → win probability conversion
# p_win = sigmoid(cp / CP_SCALE). At CP_SCALE=200, +200cp ≈ 73% win.
_CP_SCALE = 200.0


def _cp_to_wdl(cp: Optional[int], mate: Optional[int]) -> list[float]:
    """Convert Stockfish centipawn/mate score to WDL distribution.

    Args:
        cp: centipawn score (positive = white advantage), or None if mate.
        mate: mate-in-N (positive = white mates), or None if cp.

    Returns:
        [p_loss, p_draw, p_win] from white's perspective.
    """
    if mate is not None:
        if mate > 0:
            return [0.0, 0.0, 1.0]  # White mates
        elif mate < 0:
            return [1.0, 0.0, 0.0]  # Black mates
        else:
            return [0.0, 0.0, 1.0]  # Mate in 0 = already mated (shouldn't happen)

    if cp is None:
        return [1 / 3, 1 / 3, 1 / 3]  # Fallback

    # Logistic win probability
    p_win = 1.0 / (1.0 + math.exp(-cp / _CP_SCALE))
    p_loss = 1.0 - p_win
    # Inject draw probability: higher near equal positions
    # draw_factor peaks at cp=0 (~0.3) and decays with |cp|
    draw_factor = 0.3 * math.exp(-(cp / 300.0) ** 2)
    p_draw = draw_factor
    p_win = p_win * (1.0 - draw_factor)
    p_loss = p_loss * (1.0 - draw_factor)

    return [p_loss, p_draw, p_win]


def _random_position(min_ply: int = 5, max_ply: int = 15) -> Optional[chess.Board]:
    """Generate a random legal position by playing random moves."""
    board = chess.Board()
    num_ply = random.randint(min_ply, max_ply)
    for _ in range(num_ply):
        legal = list(board.legal_moves)
        if not legal or board.is_game_over():
            return None  # Dead end — discard
        board.push(random.choice(legal))
    if board.is_game_over():
        return None
    return board


def generate_sf_anchored_positions(
    stockfish_path: str,
    num_positions: int = 1000,
    depth: int = 8,
    selfplay_opening_ply: int = 10,
) -> list[GameRecord]:
    """Generate Stockfish-evaluated positions for value/policy anchoring.

    Args:
        stockfish_path: Path to Stockfish binary.
        num_positions: Total positions to generate.
        depth: Stockfish search depth.
        selfplay_opening_ply: Max ply to sample from self-play openings.

    Returns:
        List of single-position GameRecords with SF-derived targets.
    """
    n_random = num_positions // 2
    n_selfplay = num_positions - n_random

    # --- Collect positions ---
    boards: list[chess.Board] = []

    # Random legal positions
    attempts = 0
    while len(boards) < n_random and attempts < n_random * 3:
        b = _random_position()
        if b is not None:
            boards.append(b)
        attempts += 1

    # Self-play opening positions: generate random short games that mimic
    # the model's opening phase (random moves for diversity, but same depth
    # range as actual self-play openings). These positions target exactly
    # the opening territory where the model needs SF correction.
    for _ in range(n_selfplay):
        b = _random_position(min_ply=2, max_ply=selfplay_opening_ply)
        if b is not None:
            boards.append(b)

    if not boards:
        logger.warning("No positions generated for SF anchoring")
        return []

    # --- Evaluate with Stockfish ---
    records: list[GameRecord] = []
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    except Exception as e:
        logger.error("Failed to start Stockfish for anchoring: %s", e)
        return []

    try:
        for board in boards:
            try:
                info = engine.analyse(board, chess.engine.Limit(depth=depth))
            except Exception:
                continue

            score = info.get("score")
            if score is None:
                continue

            # Extract score from white's perspective
            white_score = score.white()
            cp = white_score.score()
            mate = white_score.mate()

            # WDL from white's perspective, flip if black to move
            wdl_white = _cp_to_wdl(cp, mate)
            if board.turn == chess.BLACK:
                wdl_stm = [wdl_white[2], wdl_white[1], wdl_white[0]]  # flip W/L
            else:
                wdl_stm = wdl_white

            # Best move → one-hot policy target
            best_move = info.get("pv", [None])[0]
            if best_move is None:
                continue

            policy = np.zeros(POLICY_SIZE, dtype=np.float32)
            move_idx = MoveEncoder.move_to_index(best_move)
            if move_idx < POLICY_SIZE:
                policy[move_idx] = 1.0
            else:
                continue  # Invalid move encoding
            # Mirror policy to model-space for black-to-move positions
            # (board encoding is perspective-canonical, policy target must match)
            if board.turn == chess.BLACK:
                policy = mirror_policy(policy)

            # Scalar value from WDL (for z-target compatibility)
            scalar_value = wdl_stm[2] - wdl_stm[0]  # P(win) - P(loss)

            # Legal move count for activity target
            activity = len(list(board.legal_moves)) / 40.0

            # Package as GameRecord
            record = GameRecord(
                positions=[BoardEncoder.encode_board(board)],
                policies=[torch.from_numpy(policy)],
                activities=[activity],
                root_wdl=[wdl_stm],
                surprise=[1.0],  # Neutral surprise weight
                plies=[0],
                total_plies=1,
                outcome=scalar_value,
            )
            records.append(record)
    finally:
        engine.quit()

    logger.info(
        "SF anchor: generated %d/%d positions (depth %d)",
        len(records), num_positions, depth,
    )
    return records
