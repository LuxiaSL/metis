"""Self-play game generation for AlphaZero-style training.

Two modes:
- ParallelSelfPlay: multiprocessing with centralized GPU evaluation (production)
- SelfPlayWorker: single-process batched MCTS (testing/eval)

Architecture (ParallelSelfPlay):
  Main process (GPU owner):
    - EvaluatorThread: collects leaf positions from workers, batches GPU inference
  Worker processes (CPU, ×N):
    - Each manages a few games + MCTS trees
    - Sends leaf positions for evaluation via queue
    - Receives policies/values, expands + backprops locally
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import pickle
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import chess
import numpy as np
import torch

from src.chess.board import BoardEncoder, MoveEncoder, SEQ_LEN, POLICY_SIZE
from src.chess.mcts import (
    MCTSConfig, BatchedMCTS,
    terminal_value, select_move,
)
from src.chess.mcts_array import MCTSTree

logger = logging.getLogger(__name__)


@dataclass
class GameRecord:
    """One completed self-play game."""

    positions: list[torch.Tensor] = field(default_factory=list)
    policies: list[torch.Tensor] = field(default_factory=list)
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

    def to_mcts_config(self) -> MCTSConfig:
        return MCTSConfig(
            num_simulations=self.mcts_simulations,
            cpuct=self.cpuct,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon,
            num_virtual_leaves=self.num_virtual_leaves,
        )


# ── Eval request/response protocol ────────────────────────────────────────

# Workers send: (worker_id, list[encoded_board_tensor])
# If list is empty, worker has no leaves to evaluate (all terminal).
# Evaluator sends back: (policies_ndarray, values_ndarray)
# Special sentinel: worker_id = -1 means "shutdown"

_SHUTDOWN_SENTINEL = -1


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
) -> None:
    """Worker process: manages MCTS trees for assigned games (CPU only).

    Communicates with the evaluator via queues for NN inference.
    Sends completed GameRecords to game_result_queue.
    """
    try:
        # Prevent forked workers from inheriting parent's CUDA context
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        _run_worker(
            worker_id, num_games, mcts_config, temperature,
            temperature_threshold, max_moves,
            eval_request_queue, eval_result_queue, game_result_queue,
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
) -> None:
    """Core worker loop."""
    boards = [chess.Board() for _ in range(num_games)]
    trees: list[MCTSTree] = [MCTSTree() for _ in range(num_games)]
    records = [GameRecord() for _ in range(num_games)]
    ply_counts = [0] * num_games
    active = set(range(num_games))
    cpuct = mcts_config.cpuct
    nvl = mcts_config.num_virtual_leaves

    def _request_eval(boards_to_encode: list[chess.Board]) -> tuple[np.ndarray, np.ndarray]:
        """Encode boards, send to evaluator, block for result."""
        if not boards_to_encode:
            return np.empty((0, POLICY_SIZE), dtype=np.float32), np.empty(0, dtype=np.float32)

        # Send as numpy to avoid torch tensor pickling issues
        encoded = [BoardEncoder.encode_board_array(b) for b in boards_to_encode]
        eval_request_queue.put((worker_id, encoded))
        return eval_result_queue.get()

    def _expand_roots() -> None:
        """Expand root nodes for all active games via NN evaluation."""
        active_list = sorted(active)
        if not active_list:
            return
        boards_to_eval = [boards[i] for i in active_list]
        policies, _ = _request_eval(boards_to_eval)
        for j, i in enumerate(active_list):
            trees[i].expand(trees[i].root, boards[i], policies[j])
            trees[i].add_dirichlet_noise(
                trees[i].root,
                mcts_config.dirichlet_alpha, mcts_config.dirichlet_epsilon,
            )

    # Initial expansion
    _expand_roots()

    while active:
        # ── MCTS simulations ───────────────────────────────────────
        active_list = sorted(active)

        for sim_start in range(0, mcts_config.num_simulations, nvl):
            leaves_per_game = min(nvl, mcts_config.num_simulations - sim_start)
            # Find leaves with virtual loss (parallel within each game)
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

            # Evaluate leaves.
            if leaves:
                leaf_boards = [lb for _, _, lb, _ in leaves]
                policies, values = _request_eval(leaf_boards)

                for j, (game_idx, leaf_idx, lb, path) in enumerate(leaves):
                    trees[game_idx].expand(leaf_idx, lb, policies[j])
                    trees[game_idx].remove_virtual_loss(path)
                    trees[game_idx].backup(leaf_idx, values[j])

        # ── Select moves + record positions ────────────────────────
        finished_this_round: list[int] = []

        for i in active_list:
            vc = trees[i].get_visit_counts()
            if not vc:
                records[i].outcome = _get_outcome(boards[i])
                finished_this_round.append(i)
                continue

            # Record position and MCTS policy
            records[i].positions.append(BoardEncoder.encode_board(boards[i]))
            records[i].policies.append(MoveEncoder.encode_policy(vc))

            # Select move
            ply = ply_counts[i]
            temp = temperature if ply < temperature_threshold else 0.0
            move = select_move(vc, temp)

            # Play move
            boards[i].push(move)
            ply_counts[i] += 1

            # Check termination
            if boards[i].is_game_over(claim_draw=True):
                records[i].outcome = _get_outcome(boards[i])
                finished_this_round.append(i)
            elif ply_counts[i] >= max_moves:
                records[i].outcome = 0.0
                finished_this_round.append(i)
            else:
                # Tree reuse: promote chosen child to root
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

        # Remove finished games
        for i in finished_this_round:
            active.discard(i)

        # Expand any new roots that aren't expanded (from tree reuse miss)
        unexpanded = [i for i in active if not trees[i].is_expanded[trees[i].root]]
        if unexpanded:
            boards_to_eval = [boards[i] for i in unexpanded]
            policies, _ = _request_eval(boards_to_eval)
            for j, i in enumerate(unexpanded):
                trees[i].expand(trees[i].root, boards[i], policies[j])
                trees[i].add_dirichlet_noise(
                    trees[i].root,
                    mcts_config.dirichlet_alpha, mcts_config.dirichlet_epsilon,
                )

    # Send completed game records (convert tensors to numpy for pickling)
    for record in records:
        serialized = {
            "positions": [p.numpy() for p in record.positions],
            "policies": [p.numpy() for p in record.policies],
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


# ── Evaluator (runs in main process thread) ────────────────────────────────


class _EvaluatorThread(threading.Thread):
    """Collects eval requests from workers, batches GPU inference, sends results."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        eval_request_queue: mp.Queue,
        result_queues: dict[int, mp.Queue],
        num_workers: int,
        max_batch_wait_ms: float = 5.0,
    ) -> None:
        super().__init__(daemon=True)
        self.model = model
        self.device = device
        self.eval_request_queue = eval_request_queue
        self.result_queues = result_queues
        self.num_workers = num_workers
        self.max_batch_wait = max_batch_wait_ms / 1000.0
        self._stop_event = threading.Event()
        self._total_evals = 0
        self._total_batches = 0
        self._start_time = time.monotonic()

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def stats(self) -> dict[str, float]:
        return {
            "total_evals": self._total_evals,
            "total_batches": self._total_batches,
            "avg_batch_size": (
                self._total_evals / max(self._total_batches, 1)
            ),
        }

    def run(self) -> None:
        """Main evaluator loop."""
        while not self._stop_event.is_set():
            # Collect requests from workers
            batch: list[tuple[int, list[torch.Tensor]]] = []
            try:
                # Block for first request
                req = self.eval_request_queue.get(timeout=0.5)
                if req[0] == _SHUTDOWN_SENTINEL:
                    break
                batch.append(req)

                # Greedily collect more (non-blocking)
                deadline = time.monotonic() + self.max_batch_wait
                while len(batch) < self.num_workers and time.monotonic() < deadline:
                    try:
                        req = self.eval_request_queue.get(timeout=0.001)
                        if req[0] == _SHUTDOWN_SENTINEL:
                            self._stop_event.set()
                            break
                        batch.append(req)
                    except queue.Empty:
                        break

            except queue.Empty:
                continue

            if not batch:
                continue

            # Flatten all boards from all workers (numpy → tensor)
            all_boards: list[np.ndarray] = []
            worker_sizes: list[tuple[int, int]] = []  # (worker_id, count)
            for worker_id, boards in batch:
                worker_sizes.append((worker_id, len(boards)))
                all_boards.extend(boards)

            # Evaluate
            if all_boards:
                policies, values = self._evaluate_batch(all_boards)
            else:
                policies = np.empty((0, POLICY_SIZE), dtype=np.float32)
                values = np.empty(0, dtype=np.float32)

            # Distribute results
            offset = 0
            for worker_id, count in worker_sizes:
                if count > 0:
                    self.result_queues[worker_id].put((
                        policies[offset:offset + count],
                        values[offset:offset + count],
                    ))
                    offset += count
                else:
                    # Empty request → empty response
                    self.result_queues[worker_id].put((
                        np.empty((0, POLICY_SIZE), dtype=np.float32),
                        np.empty(0, dtype=np.float32),
                    ))

            self._total_evals += len(all_boards)
            self._total_batches += 1

            # Periodic progress log (keeps Heimdall happy, shows throughput)
            if self._total_batches % 500 == 0:
                elapsed = time.monotonic() - self._start_time
                logger.info(
                    "Evaluator: %d evals, %d batches (avg %.1f/batch, %.0f evals/s)",
                    self._total_evals, self._total_batches,
                    self._total_evals / max(self._total_batches, 1),
                    self._total_evals / max(elapsed, 1e-6),
                )

    @torch.no_grad()
    def _evaluate_batch(
        self, encoded_boards: list[np.ndarray],
        max_sub_batch: int = 128,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run model inference on a batch of encoded board positions.

        Splits large batches into sub-batches to avoid OOM with AttnRes
        buffer stacking on shared GPUs.
        """
        self.model.eval()
        autocast_enabled = self.device.type == "cuda"

        if len(encoded_boards) <= max_sub_batch:
            batch = torch.from_numpy(np.stack(encoded_boards)).long().to(self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                policy_logits, values = self.model(batch)
            policies = torch.softmax(policy_logits.float(), dim=-1).cpu().numpy()
            vals = values.squeeze(-1).float().cpu().numpy()
            return policies, vals

        # Split into sub-batches for memory safety
        all_policies = []
        all_values = []
        for start in range(0, len(encoded_boards), max_sub_batch):
            sub = encoded_boards[start:start + max_sub_batch]
            batch = torch.from_numpy(np.stack(sub)).long().to(self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                policy_logits, values = self.model(batch)
            all_policies.append(torch.softmax(policy_logits.float(), dim=-1).cpu().numpy())
            all_values.append(values.squeeze(-1).float().cpu().numpy())
        return np.concatenate(all_policies), np.concatenate(all_values)


# ── Parallel self-play (production) ────────────────────────────────────────


class ParallelSelfPlay:
    """Multiprocessing self-play with centralized GPU evaluation.

    Architecture:
    - N worker processes do MCTS tree traversal on CPU
    - 1 evaluator thread in the main process handles GPU inference
    - Communication via multiprocessing queues
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

    def generate_games(self, num_games: int) -> list[GameRecord]:
        """Generate self-play games using multiprocessing.

        Uses forkserver context so workers don't inherit the parent's CUDA
        context (fork would copy GPU memory mappings into each worker).

        Args:
            num_games: Total games to generate.

        Returns:
            List of completed GameRecords.
        """
        num_workers = min(self.config.num_workers, num_games)
        games_per_worker = _distribute_games(num_games, num_workers)

        # Use forkserver: workers are forked from a clean server process
        # that doesn't have CUDA initialized, avoiding GPU memory inheritance.
        ctx = mp.get_context("forkserver")

        # Communication queues (must use same context)
        eval_request_queue: mp.Queue = ctx.Queue()
        result_queues: dict[int, mp.Queue] = {
            i: ctx.Queue() for i in range(num_workers)
        }
        game_result_queue: mp.Queue = ctx.Queue()

        # Start evaluator thread (runs in main process, has GPU access)
        evaluator = _EvaluatorThread(
            model=self.model,
            device=self.device,
            eval_request_queue=eval_request_queue,
            result_queues=result_queues,
            num_workers=num_workers,
        )
        evaluator.start()

        # Start worker processes (CPU-only, no GPU context)
        mcts_config = self.config.to_mcts_config()
        workers: list[mp.Process] = []

        for i in range(num_workers):
            p = ctx.Process(
                target=_worker_fn,
                args=(
                    i, games_per_worker[i], mcts_config,
                    self.config.temperature, self.config.temperature_threshold,
                    self.config.max_moves,
                    eval_request_queue, result_queues[i], game_result_queue,
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
            try:
                msg_type, worker_id, payload = game_result_queue.get(timeout=3600)
            except queue.Empty:
                logger.warning("Timeout waiting for worker results (3600s)")
                break

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
                    outcome=payload["outcome"],
                )
                completed_games.append(record)
            elif msg_type == "done":
                workers_done += 1
            elif msg_type == "error":
                errors.append(f"Worker {worker_id}: {payload}")
                workers_done += 1

        # Shutdown
        eval_request_queue.put((_SHUTDOWN_SENTINEL, []))
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

    def generate_games(self, num_games: int) -> list[GameRecord]:
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
        boards = [chess.Board() for _ in range(batch_size)]
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

                ply = ply_counts[game_idx]
                temp = self.config.temperature if ply < self.config.temperature_threshold else 0.0
                move = select_move(vc, temp)

                board.push(move)
                ply_counts[game_idx] += 1

                if board.is_game_over(claim_draw=True):
                    record.outcome = _get_outcome(board)
                elif ply_counts[game_idx] >= self.config.max_moves:
                    record.outcome = 0.0
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
