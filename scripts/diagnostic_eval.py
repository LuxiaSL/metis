"""Diagnostic evaluation: value head vs pure policy play against Stockfish.

Runs two sets of games against SF d1:
  1. Normal eval (value head active — MCTS uses Q-values)
  2. Value-disabled eval (Q=0 — MCTS is purely policy-driven)

Records all games as PGN for inspection.

Usage:
    python -m scripts.diagnostic_eval --checkpoint checkpoints/selfplay/latest.pt \
        --stockfish_path ~/luxi-files/bin/stockfish --num_games 10
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import chess
import chess.engine
import chess.pgn
import torch
import io

from src.model.transformer import ChessModelConfig, ChessTransformer, MODEL_CONFIGS
from src.chess.board import BoardEncoder, MoveEncoder
from src.chess.mcts import BatchedMCTS, MCTSConfig, select_move

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


class ValueDisabledWrapper(torch.nn.Module):
    """Wraps a chess model to zero out WDL predictions (uniform 1/3 each)."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        outputs = self.model(x)
        policy_logits = outputs[0]
        # Replace WDL with uniform — makes Q=0 everywhere
        uniform_wdl = torch.full_like(outputs[1], 1.0 / 3.0)
        # Keep other heads unchanged
        return (policy_logits, uniform_wdl, *outputs[2:])


def play_game(
    model: torch.nn.Module,
    engine: chess.engine.SimpleEngine,
    sf_depth: int,
    mcts_config: MCTSConfig,
    device: torch.device,
    model_is_white: bool,
    max_moves: int = 200,
) -> tuple[str, str, list[str]]:
    """Play one game, return (result, pgn_string, move_list)."""
    board = chess.Board()
    mcts = BatchedMCTS(model=model, config=mcts_config, device=device)
    moves: list[str] = []
    positions: list[str] = []

    game = chess.pgn.Game()
    game.headers["White"] = "Metis" if model_is_white else f"Stockfish d{sf_depth}"
    game.headers["Black"] = f"Stockfish d{sf_depth}" if model_is_white else "Metis"
    node = game

    move_count = 0
    while not board.is_game_over(claim_draw=True) and move_count < max_moves:
        is_model_turn = (board.turn == chess.WHITE) == model_is_white

        if is_model_turn:
            # Model move via MCTS
            vc_list = mcts.search_batch([board])
            vc = vc_list[0]
            if not vc:
                break
            move = select_move(vc, temperature=0.0)
        else:
            # Stockfish move
            result = engine.play(board, chess.engine.Limit(depth=sf_depth))
            move = result.move

        moves.append(board.san(move))
        positions.append(board.fen())
        node = node.add_variation(move)
        board.push(move)
        move_count += 1

    # Result
    if board.is_game_over(claim_draw=True):
        result_str = board.result(claim_draw=True)
    else:
        result_str = "1/2-1/2"  # max moves
    game.headers["Result"] = result_str

    pgn_str = str(game)
    return result_str, pgn_str, moves


def run_gauntlet(
    model: torch.nn.Module,
    sf_path: str,
    sf_depth: int,
    num_games: int,
    mcts_config: MCTSConfig,
    device: torch.device,
    label: str,
) -> tuple[dict, list[str]]:
    """Run a set of games, return stats and PGNs."""
    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    stats = {"wins": 0, "draws": 0, "losses": 0}
    pgns: list[str] = []

    for i in range(num_games):
        model_is_white = (i % 2 == 0)
        result, pgn, moves = play_game(
            model, engine, sf_depth, mcts_config, device,
            model_is_white, max_moves=200,
        )

        # Score from model's perspective
        if result == "1-0":
            outcome = "wins" if model_is_white else "losses"
        elif result == "0-1":
            outcome = "losses" if model_is_white else "wins"
        else:
            outcome = "draws"

        stats[outcome] += 1
        pgns.append(pgn)

        color = "W" if model_is_white else "B"
        logger.info(
            "[%s] Game %d/%d (%s): %s in %d moves — %s",
            label, i + 1, num_games, color, outcome.rstrip("s"), len(moves), result,
        )

    engine.quit()
    return stats, pgns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--stockfish_path", type=str, default="stockfish")
    parser.add_argument("--num_games", type=int, default=10)
    parser.add_argument("--sf_depth", type=int, default=1)
    parser.add_argument("--mcts_sims", type=int, default=400)
    parser.add_argument("--model_size", type=str, default="medium")
    parser.add_argument("--attn_res_boundaries", type=str, default="0,1,3,9")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    config_kwargs = MODEL_CONFIGS[args.model_size]
    if isinstance(config_kwargs, dict):
        config = ChessModelConfig(**config_kwargs)
    else:
        config = config_kwargs
    if args.attn_res_boundaries:
        config.attn_res = True
        config.attn_res_boundaries = [int(x) for x in args.attn_res_boundaries.split(",")]
    model = ChessTransformer(config).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_state = model.state_dict()
    ckpt_state = ckpt["model"]
    filtered = {k: v for k, v in ckpt_state.items() if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered, strict=False)
    logger.info("Loaded checkpoint: %s (iter %s)", args.checkpoint, ckpt.get("iteration", "?"))
    del ckpt

    model.eval()

    mcts_config = MCTSConfig(
        num_simulations=args.mcts_sims,
        cpuct=1.25,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.0,  # No noise during eval
    )

    # --- Test 1: Normal eval (value head active) ---
    logger.info("=" * 60)
    logger.info("TEST 1: Normal eval (value head active)")
    logger.info("=" * 60)
    normal_stats, normal_pgns = run_gauntlet(
        model, args.stockfish_path, args.sf_depth,
        args.num_games, mcts_config, device, "NORMAL",
    )
    logger.info(
        "Normal: +%d =%d -%d",
        normal_stats["wins"], normal_stats["draws"], normal_stats["losses"],
    )

    # --- Test 2: Value-disabled eval (pure policy) ---
    logger.info("=" * 60)
    logger.info("TEST 2: Value-disabled eval (Q=0, pure policy)")
    logger.info("=" * 60)
    disabled_model = ValueDisabledWrapper(model)
    disabled_stats, disabled_pgns = run_gauntlet(
        disabled_model, args.stockfish_path, args.sf_depth,
        args.num_games, mcts_config, device, "NO-VALUE",
    )
    logger.info(
        "No-value: +%d =%d -%d",
        disabled_stats["wins"], disabled_stats["draws"], disabled_stats["losses"],
    )

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("SUMMARY vs SF d%d (%d games each):", args.sf_depth, args.num_games)
    logger.info(
        "  Normal (value active):   +%d =%d -%d",
        normal_stats["wins"], normal_stats["draws"], normal_stats["losses"],
    )
    logger.info(
        "  No-value (Q=0, policy):  +%d =%d -%d",
        disabled_stats["wins"], disabled_stats["draws"], disabled_stats["losses"],
    )
    logger.info("=" * 60)

    # Save PGNs
    out_dir = Path("diagnostic_output")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "normal_games.pgn", "w") as f:
        f.write("\n\n".join(normal_pgns))
    with open(out_dir / "no_value_games.pgn", "w") as f:
        f.write("\n\n".join(disabled_pgns))
    logger.info("PGNs saved to %s/", out_dir)


if __name__ == "__main__":
    main()
