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
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import chess
import numpy as np
import torch
import torch.multiprocessing as tmp

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

# Workers send: (worker_id, slot_offset, count) — 3 ints, ~24 bytes pickled
# Evaluator confirms: True (results written to shared memory at worker's slots)
# Special sentinel: worker_id = -1 means "shutdown"

_SHUTDOWN_SENTINEL = -1
SLOTS_PER_WORKER = 64  # static allocation: worker i owns [i*64 : (i+1)*64]


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
) -> None:
    """Core worker loop using shared memory for board/policy/value transfer."""
    # Numpy views of shared tensors (zero-copy, backed by /dev/shm)
    slot_base = worker_id * SLOTS_PER_WORKER
    boards_np = shared_boards.numpy()
    policies_np = shared_policies.numpy()
    values_np = shared_values.numpy()

    boards = [chess.Board() for _ in range(num_games)]
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
        while not self._stop_event.is_set():
            self._batch_ready.wait()
            self._batch_ready.clear()

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
                slices: list[torch.Tensor] = []
                for _, slot_offset, count in batch:
                    if count > 0:
                        slices.append(self.shared_boards[slot_offset:slot_offset + count])
                board_tensor = torch.cat(slices, dim=0)

                # GPU inference (releases GIL — collector runs concurrently)
                policies, values = self._evaluate_batch(board_tensor)

                # Write results back to shared memory at each worker's slots
                idx = 0
                for _, slot_offset, count in batch:
                    if count > 0:
                        self._policies_np[slot_offset:slot_offset + count] = policies[idx:idx + count]
                        self._values_np[slot_offset:slot_offset + count] = values[idx:idx + count]
                        idx += count

            # Notify all workers in this batch (GIL-bound, keep minimal)
            for worker_id, _, _ in batch:
                self.result_queues[worker_id].put(True)

            self._total_evals += total_positions
            self._total_batches += 1

            # Periodic progress log
            if self._total_batches % 500 == 0:
                elapsed = time.monotonic() - self._start_time
                logger.info(
                    "Evaluator: %d evals, %d batches (avg %.1f/batch, %.0f evals/s)",
                    self._total_evals, self._total_batches,
                    self._total_evals / max(self._total_batches, 1),
                    self._total_evals / max(elapsed, 1e-6),
                )

            self._inference_idle.set()

    @torch.no_grad()
    def _evaluate_batch(
        self,
        board_tensor: torch.Tensor,
        max_sub_batch: int = 128,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run model inference on a contiguous batch of encoded boards.

        Splits large batches into sub-batches to avoid OOM on shared GPUs.

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
                policy_logits, vals = self.model(batch)
            policies = torch.softmax(policy_logits.float(), dim=-1).cpu().numpy()
            values = vals.squeeze(-1).float().cpu().numpy()
            return policies, values

        # Sub-batch for VRAM safety
        all_policies: list[np.ndarray] = []
        all_values: list[np.ndarray] = []
        for start in range(0, n, max_sub_batch):
            sub = board_tensor[start:start + max_sub_batch].to(self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                policy_logits, vals = self.model(sub)
            all_policies.append(torch.softmax(policy_logits.float(), dim=-1).cpu().numpy())
            all_values.append(vals.squeeze(-1).float().cpu().numpy())
        return np.concatenate(all_policies), np.concatenate(all_values)


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

    def generate_games(self, num_games: int) -> list[GameRecord]:
        """Generate self-play games using multiprocessing.

        Uses forkserver context so workers don't inherit the parent's CUDA
        context. Shared memory tensors are allocated before workers spawn
        so they inherit the /dev/shm handles.

        Args:
            num_games: Total games to generate.

        Returns:
            List of completed GameRecords.
        """
        num_workers = min(self.config.num_workers, num_games)
        games_per_worker = _distribute_games(num_games, num_workers)

        # Pre-allocate shared CPU tensors (backed by /dev/shm).
        # Must be created BEFORE workers spawn so forkserver children
        # inherit the shared memory file descriptors.
        max_pending = num_workers * SLOTS_PER_WORKER
        shared_boards = torch.zeros(max_pending, SEQ_LEN, dtype=torch.long).share_memory_()
        shared_policies = torch.zeros(max_pending, POLICY_SIZE, dtype=torch.float32).share_memory_()
        shared_values = torch.zeros(max_pending, dtype=torch.float32).share_memory_()

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
