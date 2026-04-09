"""Self-play game generation for AlphaZero-style training.

Two modes:
- ParallelSelfPlay: multiprocessing with centralized GPU evaluation (production)
- SelfPlayWorker: single-process batched MCTS (testing/eval)

Architecture (ParallelSelfPlay):
  Main process (GPU owner):
    - Collector thread: drains eval request queue into staging batches
    - Inference thread: runs GPU inference, writes results to shared memory
  Worker processes (CPU, ×N):
    - Each manages a few games + MCTS trees
    - Writes encoded boards to shared memory, sends slot indices via queue
    - Reads policies/values from shared memory after inference completes

Communication uses shared memory tensors (/dev/shm) for board/policy/value
data, with only lightweight slot indices (~24 bytes) going through queues.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import chess
import numpy as np
import torch
import torch.multiprocessing as tmp

from src.chess.board import BoardEncoder, MoveEncoder, SEQ_LEN, POLICY_SIZE
from src.chess.bitboard import Board as BitboardBoard
from src.chess.mcts import (
    MCTSConfig, BatchedMCTS, GumbelConfig,
    terminal_value, select_move,
    gumbel_top_k, sequential_halving,
    compute_sigma, compute_improved_policy, select_gumbel_move,
)
from src.chess.mcts_array import MCTSTree

logger = logging.getLogger(__name__)


@dataclass
class GameRecord:
    """One completed self-play game.

    With playout cap randomization (PCR), only "training moves" (full-budget
    search) are recorded. Fast moves (policy-only, for game progression) are
    played but not stored — so len(positions) <= total plies played.
    """

    positions: list[torch.Tensor] = field(default_factory=list)
    policies: list[torch.Tensor] = field(default_factory=list)
    activities: list[float] = field(default_factory=list)
    root_wdl: list[list[float]] = field(default_factory=list)  # Full WDL probs per training move
    surprise: list[float] = field(default_factory=list)  # KL(improved || prior) per training move
    plies: list[int] = field(default_factory=list)  # actual ply when each training move was recorded
    total_plies: int = 0  # total game length (set at game end, for moves-left computation)
    outcome: float = 0.0

    def __len__(self) -> int:
        return len(self.positions)


@dataclass
class SelfPlayConfig:
    """Self-play generation configuration."""

    num_parallel: int = 64
    num_workers: int = 48
    mcts_simulations: int = 800
    cpuct: float = 1.25
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    num_virtual_leaves: int = 4
    temperature: float = 1.0
    temperature_threshold: int = 30
    max_moves: int = 200

    # Gumbel AlphaZero settings
    mcts_algorithm: str = "alphazero"  # "alphazero" or "gumbel"
    gumbel_K: int = 16
    gumbel_c_visit: float = 50.0

    # Playout cap randomization (KataGo-style)
    playout_cap_fraction: float = 1.0  # 1.0 = disabled (all moves full search)
    fast_move_sims: int = 0  # Sims for fast moves (0 = raw policy only)
    # Material adjudication threshold at max_moves
    material_adjudication_threshold: float = 3.0

    def to_mcts_config(self) -> MCTSConfig:
        return MCTSConfig(
            num_simulations=self.mcts_simulations,
            cpuct=self.cpuct,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon,
            num_virtual_leaves=self.num_virtual_leaves,
        )

    def to_gumbel_config(self) -> GumbelConfig:
        return GumbelConfig(
            num_simulations=self.mcts_simulations,
            max_K=self.gumbel_K,
            c_visit=self.gumbel_c_visit,
            cpuct=self.cpuct,
        )


# ── Eval request/response protocol ────────────────────────────────────────

# Workers send: (worker_id, slot_offset, count) — 3 ints, ~24 bytes pickled
# Evaluator confirms: True (results written to shared memory at worker's slots)
# Special sentinel: worker_id = -1 means "shutdown"

_SHUTDOWN_SENTINEL = -1
SLOTS_PER_WORKER = 64  # static allocation: worker i owns [i*64 : (i+1)*64]
                       # Max per-eval: num_virtual_leaves (8) — fits comfortably

# Piece values for material adjudication (standard centipawn / 100)
_PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}


def _material_adjudicate(board: chess.Board, threshold: float = 9.0) -> float:
    """Adjudicate based on material balance when max_moves is reached.

    Returns +1 (white wins), -1 (black wins), or 0 (draw) based on whether
    the material imbalance exceeds the threshold. Gives the value head signal
    from positions that would otherwise all be labeled as draws.
    """
    white_material = sum(
        _PIECE_VALUES[pt] * len(board.pieces(pt, chess.WHITE))
        for pt in _PIECE_VALUES
    )
    black_material = sum(
        _PIECE_VALUES[pt] * len(board.pieces(pt, chess.BLACK))
        for pt in _PIECE_VALUES
    )
    diff = white_material - black_material
    if diff > threshold:
        return 1.0
    elif diff < -threshold:
        return -1.0
    return 0.0


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax over the last axis (numerically stable)."""
    x = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _legal_move_count(board: chess.Board) -> int:
    if isinstance(board, BitboardBoard):
        return board.legal_move_count()
    return len(list(board.legal_moves))


# ── Worker process ─────────────────────────────────────────────────────────


def _worker_fn(
    worker_id: int,
    num_games: int,
    mcts_config: MCTSConfig,
    temperature: float,
    temperature_threshold: int,
    max_moves: int,
    eval_request_queue: mp.Queue,
    eval_result_queue: mp.Queue,
    game_result_queue: mp.Queue,
    shared_boards: torch.Tensor,
    shared_policies: torch.Tensor,
    shared_values: torch.Tensor,
    mcts_algorithm: str = "alphazero",
    gumbel_config: Optional[GumbelConfig] = None,
    playout_cap_fraction: float = 1.0,
    material_adjudication_threshold: float = 9.0,
    fast_move_sims: int = 0,
) -> None:
    """Worker process: manages MCTS trees for assigned games (CPU only).

    Communicates with the evaluator via shared memory + lightweight queue messages.
    Sends completed GameRecords to game_result_queue.
    """
    try:
        # Prevent forked workers from inheriting parent's CUDA context
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        _run_worker(
            worker_id, num_games, mcts_config, temperature,
            temperature_threshold, max_moves,
            eval_request_queue, eval_result_queue, game_result_queue,
            shared_boards, shared_policies, shared_values,
            mcts_algorithm=mcts_algorithm,
            gumbel_config=gumbel_config,
            playout_cap_fraction=playout_cap_fraction,
            material_adjudication_threshold=material_adjudication_threshold,
            fast_move_sims=fast_move_sims,
        )
    except Exception as e:
        logger.error("Worker %d crashed: %s", worker_id, e, exc_info=True)
        # Send error marker so main process doesn't hang
        game_result_queue.put(("error", worker_id, str(e)))


def _run_worker(
    worker_id: int,
    num_games: int,
    mcts_config: MCTSConfig,
    temperature: float,
    temperature_threshold: int,
    max_moves: int,
    eval_request_queue: mp.Queue,
    eval_result_queue: mp.Queue,
    game_result_queue: mp.Queue,
    shared_boards: torch.Tensor,
    shared_policies: torch.Tensor,
    shared_values: torch.Tensor,
    mcts_algorithm: str = "alphazero",
    gumbel_config: Optional[GumbelConfig] = None,
    playout_cap_fraction: float = 1.0,
    material_adjudication_threshold: float = 9.0,
    fast_move_sims: int = 0,
) -> None:
    """Core worker loop using shared memory for board/policy/value transfer."""
    # Numpy views of shared tensors (zero-copy, backed by /dev/shm)
    slot_base = worker_id * SLOTS_PER_WORKER
    boards_np = shared_boards.numpy()
    policies_np = shared_policies.numpy()
    values_np = shared_values.numpy()

    boards = [BitboardBoard() for _ in range(num_games)]
    trees: list[MCTSTree] = [MCTSTree() for _ in range(num_games)]
    records = [GameRecord() for _ in range(num_games)]
    ply_counts = [0] * num_games
    active = set(range(num_games))
    cpuct = mcts_config.cpuct
    nvl = mcts_config.num_virtual_leaves

    def _request_eval(boards_to_encode: list[chess.Board]) -> tuple[np.ndarray, np.ndarray]:
        """Encode boards into shared memory, request eval, read results back."""
        count = len(boards_to_encode)
        if count == 0:
            return np.empty((0, POLICY_SIZE), dtype=np.float32), np.empty(0, dtype=np.float32)

        # Write encoded boards into our slot range in shared memory
        for i, b in enumerate(boards_to_encode):
            BoardEncoder._encode_into(b, boards_np[slot_base + i])

        # Send lightweight slot index message (3 ints, ~24 bytes pickled)
        eval_request_queue.put((worker_id, slot_base, count))

        # Block until evaluator confirms results are written to our slots
        eval_result_queue.get()

        # Read results directly from shared memory (no copy needed —
        # our slots won't be overwritten until we send the next request)
        return policies_np[slot_base:slot_base + count], values_np[slot_base:slot_base + count]

    use_gumbel = mcts_algorithm == "gumbel" and gumbel_config is not None
    # Per-game storage for raw logits + root WDL (needed by Gumbel)
    root_logits: list[Optional[np.ndarray]] = [None] * num_games
    root_values: list[float] = [0.0] * num_games  # scalar for MCTS backup
    root_wdl: list[Optional[np.ndarray]] = [None] * num_games  # full WDL for training targets

    def _expand_roots() -> None:
        """Expand root nodes for all active games via NN evaluation."""
        active_list = sorted(active)
        if not active_list:
            return
        boards_to_eval = [boards[i] for i in active_list]
        logits_batch, wdl_batch = _request_eval(boards_to_eval)
        for j, i in enumerate(active_list):
            # Store raw logits and WDL for Gumbel path
            root_logits[i] = logits_batch[j].copy()
            root_wdl[i] = wdl_batch[j].copy()  # (3,) WDL probs
            root_values[i] = float(wdl_batch[j][2] - wdl_batch[j][0])  # scalar for MCTS
            # Softmax for tree expansion priors
            probs = _softmax(logits_batch[j:j+1])[0]
            trees[i].expand(trees[i].root, boards[i], probs)
            trees[i].add_dirichlet_noise(
                trees[i].root,
                mcts_config.dirichlet_alpha, mcts_config.dirichlet_epsilon,
            )

    # Initial expansion
    _expand_roots()

    def _eval_vl_batch(boards_to_eval: list) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate a list of boards via the shared evaluator.

        Returns (softmax_policies, scalar_values). Used by sequential_halving
        with num_virtual_leaves > 1 for batched leaf evaluation.
        """
        if not boards_to_eval:
            return np.empty((0, POLICY_SIZE), dtype=np.float32), np.empty(0, dtype=np.float32)
        logits_batch, wdl_batch = _request_eval(boards_to_eval)
        values = wdl_batch[:, 2] - wdl_batch[:, 0]  # WDL→scalar for MCTS backup
        return _softmax(logits_batch), values

    # Track selected moves per game for tree reuse
    selected_moves: dict[int, Optional[chess.Move]] = {}

    while active:
        active_list = sorted(active)
        selected_moves.clear()

        if use_gumbel:
            # ── Gumbel AlphaZero search (per-game, VL-batched evals) ──
            # With playout cap randomization (PCR): only a fraction of moves
            # get full Sequential Halving search. The rest play from the raw
            # network policy for game progression, without recording training data.
            assert gumbel_config is not None
            for i in active_list:
                logits_i = root_logits[i]
                if logits_i is None:
                    selected_moves[i] = None
                    continue

                legal_moves = list(boards[i].legal_moves)
                if not legal_moves:
                    records[i].outcome = _get_outcome(boards[i])
                    selected_moves[i] = None
                    continue

                is_training_move = (random.random() < playout_cap_fraction)
                ply = ply_counts[i]

                if is_training_move:
                    # ── Full search: Gumbel SH → record training data ──
                    K = min(gumbel_config.max_K, len(legal_moves))
                    legal_indices = [MoveEncoder.move_to_index(m) for m in legal_moves]
                    selected_indices, gumbel_scores = gumbel_top_k(logits_i, legal_indices, K)

                    idx_to_move = {MoveEncoder.move_to_index(m): m for m in legal_moves}
                    candidate_moves = [idx_to_move[si] for si in selected_indices]

                    winner = sequential_halving(
                        trees[i], boards[i], candidate_moves,
                        gumbel_config.num_simulations, _eval_vl_batch,
                        gumbel_config.cpuct,
                        num_virtual_leaves=nvl,
                    )

                    completed_q = trees[i].get_completed_q_values(
                        root_values[i], legal_moves,
                    )
                    sigma = compute_sigma(completed_q, gumbel_config.c_visit)
                    improved_policy = compute_improved_policy(
                        logits_i, completed_q, legal_moves, sigma,
                    )

                    # Compute policy surprise: KL(improved || prior) over legal moves
                    legal_indices_arr = np.array(legal_indices, dtype=np.intp)
                    prior_logits = logits_i[legal_indices_arr]
                    prior_probs = _softmax(prior_logits[np.newaxis])[0]
                    improved_legal = improved_policy[legal_indices_arr]
                    # KL = sum(improved * log(improved / prior)) with epsilon for stability
                    _eps = 1e-8
                    kl_div = float(np.sum(
                        improved_legal * np.log((improved_legal + _eps) / (prior_probs + _eps))
                    ))
                    kl_div = max(kl_div, 0.0)  # Clamp numerical noise

                    # Record training data (position, improved policy, root value, surprise, ply)
                    records[i].positions.append(BoardEncoder.encode_board(boards[i]))
                    records[i].policies.append(torch.from_numpy(improved_policy))
                    records[i].activities.append(_legal_move_count(boards[i]) / 40.0)
                    records[i].root_wdl.append(root_wdl[i].tolist())
                    records[i].surprise.append(kl_div)
                    records[i].plies.append(ply)

                    # Select move via Gumbel
                    if ply < temperature_threshold:
                        q_values = trees[i].get_child_q_values(negate=True)
                        move = select_gumbel_move(
                            candidate_moves, gumbel_scores, q_values, sigma,
                        )
                    else:
                        move = winner
                else:
                    # ── Fast move: no training data recorded ──
                    legal_indices = np.array(
                        [MoveEncoder.move_to_index(m) for m in legal_moves],
                        dtype=np.intp,
                    )

                    if fast_move_sims > 0:
                        # Lightweight Gumbel search for better game progression
                        K = min(gumbel_config.max_K, len(legal_moves))
                        selected_indices, gumbel_scores = gumbel_top_k(
                            logits_i, legal_indices.tolist(), K,
                        )
                        idx_to_move = {MoveEncoder.move_to_index(m): m for m in legal_moves}
                        candidate_moves = [idx_to_move[si] for si in selected_indices]

                        sequential_halving(
                            trees[i], boards[i], candidate_moves,
                            fast_move_sims, _eval_vl_batch,
                            gumbel_config.cpuct,
                            num_virtual_leaves=nvl,
                        )

                        completed_q = trees[i].get_completed_q_values(
                            root_values[i], legal_moves,
                        )
                        sigma = compute_sigma(completed_q, gumbel_config.c_visit)

                        if ply < temperature_threshold:
                            q_values = trees[i].get_child_q_values(negate=True)
                            move = select_gumbel_move(
                                candidate_moves, gumbel_scores, q_values, sigma,
                            )
                        else:
                            # After temperature threshold, pick best improved policy
                            improved = compute_improved_policy(
                                logits_i, completed_q, legal_moves, sigma,
                            )
                            # improved is 4672-dim; find which legal move has max probability
                            best_policy_idx = int(np.argmax(improved[legal_indices]))
                            move = legal_moves[best_policy_idx]
                    else:
                        # Raw policy only (0 sims)
                        legal_logits = logits_i[legal_indices]

                        if ply < temperature_threshold:
                            probs = _softmax(legal_logits[np.newaxis])[0]
                        else:
                            # Soft temperature (τ=0.5) instead of argmax to maintain
                            # game diversity across fast moves in the 75% PCR path
                            scaled = legal_logits * 2.0  # 1/τ = 1/0.5 = 2
                            probs = _softmax(scaled[np.newaxis])[0]
                        move_idx = np.random.choice(len(legal_moves), p=probs)
                        move = legal_moves[move_idx]

                boards[i].push(move)
                ply_counts[i] += 1
                selected_moves[i] = move

        else:
            # ── Standard AlphaZero MCTS ───────────────────────────
            for sim_start in range(0, mcts_config.num_simulations, nvl):
                leaves_per_game = min(nvl, mcts_config.num_simulations - sim_start)
                leaves: list[tuple[int, int, chess.Board, np.ndarray]] = []

                for i in active_list:
                    for _ in range(leaves_per_game):
                        leaf_idx, search_board, path = trees[i].find_leaf_with_virtual_loss(
                            boards[i], cpuct,
                        )
                        if trees[i].is_terminal[leaf_idx] or search_board.is_game_over(claim_draw=False):
                            trees[i].remove_virtual_loss(path)
                            trees[i].backup(leaf_idx, terminal_value(search_board))
                        else:
                            leaves.append((i, leaf_idx, search_board, path))

                if leaves:
                    leaf_boards = [lb for _, _, lb, _ in leaves]
                    logits_batch, values = _request_eval(leaf_boards)

                    for j, (game_idx, leaf_idx, lb, path) in enumerate(leaves):
                        probs = _softmax(logits_batch[j:j+1])[0]
                        trees[game_idx].expand(leaf_idx, lb, probs)
                        trees[game_idx].remove_virtual_loss(path)
                        trees[game_idx].backup(leaf_idx, values[j])

            # Select moves for AlphaZero path
            for i in active_list:
                vc = trees[i].get_visit_counts()
                if not vc:
                    records[i].outcome = _get_outcome(boards[i])
                    selected_moves[i] = None
                    continue

                records[i].positions.append(BoardEncoder.encode_board(boards[i]))
                records[i].policies.append(MoveEncoder.encode_policy(vc))
                records[i].activities.append(_legal_move_count(boards[i]) / 40.0)
                records[i].plies.append(ply_counts[i])

                ply = ply_counts[i]
                temp = temperature if ply < temperature_threshold else 0.0
                move = select_move(vc, temp)

                boards[i].push(move)
                ply_counts[i] += 1
                selected_moves[i] = move

        # ── Check termination + tree management (both algorithms) ──
        finished_this_round: list[int] = []

        for i in active_list:
            move = selected_moves.get(i)
            if move is None:
                # No valid move — game already marked as finished above
                records[i].total_plies = ply_counts[i]
                finished_this_round.append(i)
            elif boards[i].is_game_over(claim_draw=True):
                records[i].outcome = _get_outcome(boards[i])
                records[i].total_plies = ply_counts[i]
                finished_this_round.append(i)
            elif ply_counts[i] >= max_moves:
                records[i].outcome = _material_adjudicate(
                    boards[i], threshold=material_adjudication_threshold,
                )
                records[i].total_plies = ply_counts[i]
                finished_this_round.append(i)
            else:
                # Tree reuse (AlphaZero) or fresh tree (Gumbel)
                if use_gumbel:
                    trees[i].reset()
                else:
                    child_idx = trees[i].get_child_for_move(move)
                    if child_idx is not None and trees[i].remaining_capacity() > 20_000:
                        was_expanded = bool(trees[i].is_expanded[child_idx])
                        trees[i].reroot(child_idx)
                        if was_expanded:
                            trees[i].add_dirichlet_noise(
                                trees[i].root,
                                mcts_config.dirichlet_alpha,
                                mcts_config.dirichlet_epsilon,
                            )
                    else:
                        trees[i].reset()

        for i in finished_this_round:
            active.discard(i)

        # Expand any new roots that aren't expanded
        unexpanded = [i for i in active if not trees[i].is_expanded[trees[i].root]]
        if unexpanded:
            boards_to_eval = [boards[i] for i in unexpanded]
            logits_batch, wdl_batch = _request_eval(boards_to_eval)
            for j, i in enumerate(unexpanded):
                root_logits[i] = logits_batch[j].copy()
                root_wdl[i] = wdl_batch[j].copy()
                root_values[i] = float(wdl_batch[j][2] - wdl_batch[j][0])
                probs = _softmax(logits_batch[j:j+1])[0]
                trees[i].expand(trees[i].root, boards[i], probs)
                trees[i].add_dirichlet_noise(
                    trees[i].root,
                    mcts_config.dirichlet_alpha, mcts_config.dirichlet_epsilon,
                )

    # Send completed game records (convert tensors to numpy for pickling)
    for record in records:
        serialized = {
            "positions": [p.numpy() for p in record.positions],
            "policies": [p.numpy() for p in record.policies],
            "activities": record.activities,
            "root_wdl": record.root_wdl,
            "surprise": record.surprise,
            "plies": record.plies,
            "total_plies": record.total_plies,
            "outcome": record.outcome,
        }
        game_result_queue.put(("game", worker_id, serialized))

    # Signal worker is done
    game_result_queue.put(("done", worker_id, None))


def _get_outcome(board: chess.Board) -> float:
    result = board.result(claim_draw=True)
    if result == "1-0":
        return 1.0
    elif result == "0-1":
        return -1.0
    return 0.0


# ── Double-buffered evaluator (runs in main process) ─────────────────────


class _DoubleBufferedEvaluator:
    """Two-thread evaluator: collector + inference for pipelined GPU batching.

    Collector thread drains the request queue while inference processes the
    previous batch on GPU. GPU inference releases the GIL, so the collector
    runs concurrently during CUDA kernels.

    All board/policy/value data flows through shared memory tensors —
    only lightweight slot indices (3 ints per request) go through queues.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        eval_request_queue: mp.Queue,
        result_queues: dict[int, mp.Queue],
        num_workers: int,
        shared_boards: torch.Tensor,
        shared_policies: torch.Tensor,
        shared_values: torch.Tensor,
        max_batch_wait_ms: float = 5.0,
    ) -> None:
        self.model = model
        self.device = device
        self.eval_request_queue = eval_request_queue
        self.result_queues = result_queues
        self.num_workers = num_workers
        self.max_batch_wait = max_batch_wait_ms / 1000.0

        # Shared memory (numpy views for fast read/write in dispatch)
        self.shared_boards = shared_boards
        self._policies_np = shared_policies.numpy()
        self._values_np = shared_values.numpy()

        # Double-buffer synchronization
        self._stop_event = threading.Event()
        self._batch: list[tuple[int, int, int]] = []  # handed from collector
        self._batch_ready = threading.Event()
        self._inference_idle = threading.Event()
        self._inference_idle.set()  # inference starts idle

        # Stats (written by inference thread only, read from main after join)
        self._total_evals = 0
        self._total_batches = 0
        self._start_time = time.monotonic()

        self._collector = threading.Thread(
            target=self._collector_loop, name="eval-collector", daemon=True,
        )
        self._inference = threading.Thread(
            target=self._inference_loop, name="eval-inference", daemon=True,
        )

    def start(self) -> None:
        self._collector.start()
        self._inference.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._batch_ready.set()  # wake inference thread if blocked

    def join(self, timeout: Optional[float] = None) -> None:
        self._collector.join(timeout=timeout)
        self._inference.join(timeout=timeout)

    @property
    def stats(self) -> dict[str, float]:
        return {
            "total_evals": self._total_evals,
            "total_batches": self._total_batches,
            "avg_batch_size": (
                self._total_evals / max(self._total_batches, 1)
            ),
        }

    # ── Collector thread ─────────────────────────────────────────────

    def _collector_loop(self) -> None:
        """Drain request queue, accumulate batches, hand off to inference."""
        staging: list[tuple[int, int, int]] = []

        while not self._stop_event.is_set():
            # Block for first request
            try:
                req = self.eval_request_queue.get(timeout=0.5)
                if req[0] == _SHUTDOWN_SENTINEL:
                    self._stop_event.set()
                    self._batch_ready.set()
                    break
                staging.append(req)
            except queue.Empty:
                continue

            # Greedily collect more (non-blocking) until batch threshold or timeout
            deadline = time.monotonic() + self.max_batch_wait
            while len(staging) < self.num_workers and time.monotonic() < deadline:
                try:
                    req = self.eval_request_queue.get(timeout=0.001)
                    if req[0] == _SHUTDOWN_SENTINEL:
                        self._stop_event.set()
                        self._batch_ready.set()
                        break
                    staging.append(req)
                except queue.Empty:
                    break

            if not staging:
                continue

            # Wait for inference thread to finish previous batch
            self._inference_idle.wait()
            if self._stop_event.is_set():
                # Drain: send confirmations so workers don't hang
                for worker_id, _, _ in staging:
                    try:
                        self.result_queues[worker_id].put(True)
                    except Exception:
                        pass
                break

            # Hand off batch to inference thread (swap)
            self._batch = staging
            staging = []
            self._inference_idle.clear()
            self._batch_ready.set()

    # ── Inference thread ─────────────────────────────────────────────

    def _inference_loop(self) -> None:
        """Process batches on GPU, write results to shared memory."""
        # Timing accumulators for profiling (reset every 500 batches)
        _t_gather = 0.0
        _t_inference = 0.0
        _t_writeback = 0.0
        _t_notify = 0.0
        _t_wait = 0.0
        _prof_batches = 0

        while not self._stop_event.is_set():
            _tw0 = time.monotonic()
            self._batch_ready.wait()
            self._batch_ready.clear()
            _t_wait += time.monotonic() - _tw0

            if self._stop_event.is_set():
                break

            batch = self._batch
            self._batch = []

            if not batch:
                self._inference_idle.set()
                continue

            total_positions = sum(count for _, _, count in batch)

            if total_positions > 0:
                # Gather boards from shared memory into a contiguous tensor
                _tg0 = time.monotonic()
                slices: list[torch.Tensor] = []
                for _, slot_offset, count in batch:
                    if count > 0:
                        slices.append(self.shared_boards[slot_offset:slot_offset + count])
                board_tensor = torch.cat(slices, dim=0)
                _t_gather += time.monotonic() - _tg0

                # GPU inference (releases GIL — collector runs concurrently)
                _ti0 = time.monotonic()
                policies, values = self._evaluate_batch(board_tensor)
                _t_inference += time.monotonic() - _ti0

                # Write results back to shared memory at each worker's slots
                _tw0 = time.monotonic()
                idx = 0
                for _, slot_offset, count in batch:
                    if count > 0:
                        self._policies_np[slot_offset:slot_offset + count] = policies[idx:idx + count]
                        self._values_np[slot_offset:slot_offset + count] = values[idx:idx + count]
                        idx += count
                _t_writeback += time.monotonic() - _tw0

            # Notify all workers in this batch (GIL-bound, keep minimal)
            _tn0 = time.monotonic()
            for worker_id, _, _ in batch:
                self.result_queues[worker_id].put(True)
            _t_notify += time.monotonic() - _tn0

            self._total_evals += total_positions
            self._total_batches += 1
            _prof_batches += 1

            # Periodic progress log with timing breakdown
            if self._total_batches % 500 == 0:
                elapsed = time.monotonic() - self._start_time
                total_prof = _t_gather + _t_inference + _t_writeback + _t_notify + _t_wait
                if total_prof > 0 and _prof_batches > 0:
                    logger.info(
                        "Evaluator: %d evals, %d batches (avg %.1f/batch, %.0f evals/s) "
                        "| wait %.0f%% gather %.0f%% infer %.0f%% write %.0f%% notify %.0f%%",
                        self._total_evals, self._total_batches,
                        self._total_evals / max(self._total_batches, 1),
                        self._total_evals / max(elapsed, 1e-6),
                        _t_wait / total_prof * 100,
                        _t_gather / total_prof * 100,
                        _t_inference / total_prof * 100,
                        _t_writeback / total_prof * 100,
                        _t_notify / total_prof * 100,
                    )
                else:
                    logger.info(
                        "Evaluator: %d evals, %d batches (avg %.1f/batch, %.0f evals/s)",
                        self._total_evals, self._total_batches,
                        self._total_evals / max(self._total_batches, 1),
                        self._total_evals / max(elapsed, 1e-6),
                    )
                _t_gather = _t_inference = _t_writeback = _t_notify = _t_wait = 0.0
                _prof_batches = 0

            self._inference_idle.set()

    @torch.no_grad()
    def _evaluate_batch(
        self,
        board_tensor: torch.Tensor,
        max_sub_batch: int = 4096,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run model inference on a contiguous batch of encoded boards.

        Splits large batches into sub-batches if they exceed max_sub_batch.

        Args:
            board_tensor: (N, 67) long tensor on CPU from shared memory.

        Returns:
            (policies, values) as numpy arrays.
        """
        self.model.eval()
        autocast_enabled = self.device.type == "cuda"
        n = board_tensor.shape[0]

        if n <= max_sub_batch:
            batch = board_tensor.to(self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                policy_logits, wdl_logits, *_ = self.model(batch)
            # Return raw logits (workers compute softmax locally when needed).
            # This enables Gumbel search which needs raw logits, and doesn't
            # hurt AlphaZero since expand() normalizes priors anyway.
            logits = policy_logits.float().cpu().numpy()
            wdl_probs = torch.softmax(wdl_logits.float(), dim=-1).cpu().numpy()
            return logits, wdl_probs

        # Sub-batch fallback for very large batches
        all_logits: list[np.ndarray] = []
        all_wdl: list[np.ndarray] = []
        for start in range(0, n, max_sub_batch):
            sub = board_tensor[start:start + max_sub_batch].to(self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                policy_logits, wdl_logits, *_ = self.model(sub)
            all_logits.append(policy_logits.float().cpu().numpy())
            all_wdl.append(torch.softmax(wdl_logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(all_logits), np.concatenate(all_wdl)


# ── Parallel self-play (production) ────────────────────────────────────────


class ParallelSelfPlay:
    """Multiprocessing self-play with centralized GPU evaluation.

    Architecture:
    - N worker processes do MCTS tree traversal on CPU
    - Double-buffered evaluator in main process: collector + inference threads
    - Communication via shared memory tensors + lightweight queue messages
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: SelfPlayConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device

    def generate_games(
        self,
        num_games: int,
        shutdown_event: Optional[threading.Event] = None,
    ) -> list[GameRecord]:
        """Generate self-play games using multiprocessing.

        Uses forkserver context so workers don't inherit the parent's CUDA
        context. Shared memory tensors are allocated before workers spawn
        so they inherit the /dev/shm handles.

        Args:
            num_games: Total games to generate.
            shutdown_event: If set, abort self-play early for graceful shutdown.

        Returns:
            List of completed GameRecords (may be partial on shutdown).
        """
        num_workers = min(self.config.num_workers, num_games)
        games_per_worker = _distribute_games(num_games, num_workers)

        # Pre-allocate shared CPU tensors (backed by /dev/shm).
        # Must be created BEFORE workers spawn so forkserver children
        # inherit the shared memory file descriptors.
        max_pending = num_workers * SLOTS_PER_WORKER
        shared_boards = torch.zeros(max_pending, SEQ_LEN, dtype=torch.long).share_memory_()
        shared_policies = torch.zeros(max_pending, POLICY_SIZE, dtype=torch.float32).share_memory_()
        shared_values = torch.zeros(max_pending, 3, dtype=torch.float32).share_memory_()  # WDL probs

        shm_mb = (
            shared_boards.numel() * shared_boards.element_size()
            + shared_policies.numel() * shared_policies.element_size()
            + shared_values.numel() * shared_values.element_size()
        ) / 1e6
        logger.info(
            "Shared memory: %d workers × %d slots, %.1f MB total (/dev/shm)",
            num_workers, SLOTS_PER_WORKER, shm_mb,
        )

        # Use torch.multiprocessing context for correct tensor pickling
        ctx = tmp.get_context("forkserver")

        # Communication queues
        eval_request_queue = ctx.Queue()
        result_queues: dict[int, mp.Queue] = {
            i: ctx.Queue() for i in range(num_workers)
        }
        game_result_queue = ctx.Queue()

        # Start double-buffered evaluator (runs in main process, has GPU)
        evaluator = _DoubleBufferedEvaluator(
            model=self.model,
            device=self.device,
            eval_request_queue=eval_request_queue,
            result_queues=result_queues,
            num_workers=num_workers,
            shared_boards=shared_boards,
            shared_policies=shared_policies,
            shared_values=shared_values,
        )
        evaluator.start()

        # Start worker processes (CPU-only, no GPU context)
        mcts_config = self.config.to_mcts_config()
        gumbel_cfg = self.config.to_gumbel_config() if self.config.mcts_algorithm == "gumbel" else None
        workers: list[mp.Process] = []

        for i in range(num_workers):
            p = ctx.Process(
                target=_worker_fn,
                args=(
                    i, games_per_worker[i], mcts_config,
                    self.config.temperature, self.config.temperature_threshold,
                    self.config.max_moves,
                    eval_request_queue, result_queues[i], game_result_queue,
                    shared_boards, shared_policies, shared_values,
                    self.config.mcts_algorithm, gumbel_cfg,
                    self.config.playout_cap_fraction,
                    self.config.material_adjudication_threshold,
                    self.config.fast_move_sims,
                ),
                daemon=True,
            )
            p.start()
            workers.append(p)

        # Collect results
        completed_games: list[GameRecord] = []
        workers_done = 0
        errors: list[str] = []

        while workers_done < num_workers:
            # Use short timeout so we can check shutdown_event frequently
            try:
                msg_type, worker_id, payload = game_result_queue.get(timeout=2.0)
            except queue.Empty:
                if shutdown_event is not None and shutdown_event.is_set():
                    logger.info("Self-play interrupted by shutdown (%d/%d games collected)",
                                len(completed_games), num_games)
                    break
                continue

            if msg_type == "game":
                if len(completed_games) % 10 == 0 and len(completed_games) > 0:
                    logger.info(
                        "Self-play progress: %d/%d games complete",
                        len(completed_games), num_games,
                    )
                # Reconstruct GameRecord from numpy-serialized data
                record = GameRecord(
                    positions=[torch.from_numpy(p) for p in payload["positions"]],
                    policies=[torch.from_numpy(p) for p in payload["policies"]],
                    activities=payload.get("activities", []),
                    root_wdl=payload.get("root_wdl", []),
                    surprise=payload.get("surprise", []),
                    plies=payload.get("plies", []),
                    total_plies=payload.get("total_plies", 0),
                    outcome=payload["outcome"],
                )
                completed_games.append(record)
            elif msg_type == "done":
                workers_done += 1
            elif msg_type == "error":
                errors.append(f"Worker {worker_id}: {payload}")
                workers_done += 1

        # Shutdown
        eval_request_queue.put((_SHUTDOWN_SENTINEL, 0, 0))
        evaluator.stop()
        evaluator.join(timeout=5)

        for p in workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()

        if errors:
            logger.error("Worker errors: %s", errors)

        eval_stats = evaluator.stats
        logger.info(
            "Self-play: %d games, %d NN evals in %d batches (avg %.1f/batch)",
            len(completed_games),
            int(eval_stats["total_evals"]),
            int(eval_stats["total_batches"]),
            eval_stats["avg_batch_size"],
        )

        return completed_games


# ── Single-process fallback ────────────────────────────────────────────────


class SelfPlayWorker:
    """Single-process self-play using BatchedMCTS. For testing/eval."""

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
            model=model, config=config.to_mcts_config(), device=device,
        )

    def generate_games(
        self,
        num_games: int,
        shutdown_event: Optional[threading.Event] = None,
    ) -> list[GameRecord]:
        completed: list[GameRecord] = []
        remaining = num_games

        while remaining > 0:
            if shutdown_event is not None and shutdown_event.is_set():
                break
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
        boards = [BitboardBoard() for _ in range(batch_size)]
        records = [GameRecord() for _ in range(batch_size)]
        active = list(range(batch_size))
        ply_counts = [0] * batch_size

        while active:
            active_boards = [boards[i] for i in active]
            visit_counts_list = self.mcts.search_batch(active_boards)

            next_active: list[int] = []
            for j, game_idx in enumerate(active):
                vc = visit_counts_list[j]
                board = boards[game_idx]
                record = records[game_idx]

                if not vc:
                    record.outcome = _get_outcome(board)
                    continue

                record.positions.append(BoardEncoder.encode_board(board))
                record.policies.append(MoveEncoder.encode_policy(vc))
                record.activities.append(_legal_move_count(board) / 40.0)
                record.root_wdl.append([1/3, 1/3, 1/3])  # Single-process path: uniform WDL fallback
                record.surprise.append(0.0)  # No surprise data without Gumbel search
                record.plies.append(ply_counts[game_idx])

                ply = ply_counts[game_idx]
                temp = self.config.temperature if ply < self.config.temperature_threshold else 0.0
                move = select_move(vc, temp)

                board.push(move)
                ply_counts[game_idx] += 1

                if board.is_game_over(claim_draw=True):
                    record.outcome = _get_outcome(board)
                    record.total_plies = ply_counts[game_idx]
                elif ply_counts[game_idx] >= self.config.max_moves:
                    record.outcome = _material_adjudicate(
                        board, threshold=self.config.material_adjudication_threshold,
                    )
                    record.total_plies = ply_counts[game_idx]
                else:
                    next_active.append(game_idx)

            active = next_active

        return records


# ── Helpers ────────────────────────────────────────────────────────────────


def _distribute_games(total: int, num_workers: int) -> list[int]:
    """Distribute games evenly across workers."""
    base = total // num_workers
    remainder = total % num_workers
    return [base + (1 if i < remainder else 0) for i in range(num_workers)]
