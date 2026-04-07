"""Array-based MCTS tree with Numba JIT for hot-path traversal.

Replaces per-node MCTSNode objects with struct-of-arrays layout.
Tree traversal (select/find_leaf/backup) compiled via Numba njit (~1us),
board ops (legal_moves, push, is_game_over) stay in Python, called once
at the leaf after path selection.
"""

from __future__ import annotations

import math
from typing import Optional

import chess
import numba
import numpy as np

from src.chess.board import MoveEncoder

DEFAULT_CAPACITY = 300_000
_MAX_DEPTH = 256


# ── Numba JIT functions ──────────────────────────────────────────────────


@numba.njit(cache=True)
def _select_child_jit(
    node: int,
    children_start: np.ndarray,
    num_children: np.ndarray,
    visit_count: np.ndarray,
    value_sum: np.ndarray,
    prior: np.ndarray,
    cpuct: float,
) -> int:
    """Select child with highest PUCT score. Returns child node index."""
    start = children_start[node]
    nc = num_children[node]
    parent_visits = visit_count[node]
    if parent_visits < 1:
        parent_visits = 1
    exploration_scale = cpuct * math.sqrt(parent_visits)

    best_score = -1e30
    best_child = start

    for i in range(nc):
        child = start + i
        visits = visit_count[child]
        if visits > 0:
            q = value_sum[child] / visits
        else:
            q = 0.0
        score = q + exploration_scale * prior[child] / (1 + visits)
        if score > best_score:
            best_score = score
            best_child = child

    return best_child


@numba.njit(cache=True)
def _find_leaf_jit(
    root: int,
    children_start: np.ndarray,
    num_children: np.ndarray,
    visit_count: np.ndarray,
    value_sum: np.ndarray,
    prior: np.ndarray,
    is_expanded: np.ndarray,
    is_terminal: np.ndarray,
    cpuct: float,
    vl: float,
    path_buf: np.ndarray,
) -> int:
    """Traverse to leaf with virtual loss. Returns path length.

    path_buf[:return] has child indices visited (excluding root).
    """
    node = root
    path_len = 0

    while is_expanded[node] and not is_terminal[node]:
        child = _select_child_jit(
            node, children_start, num_children,
            visit_count, value_sum, prior, cpuct,
        )
        visit_count[child] += 1
        value_sum[child] -= vl
        path_buf[path_len] = child
        path_len += 1
        node = child

    return path_len


@numba.njit(cache=True)
def _backup_jit(
    leaf: int,
    value: float,
    parent: np.ndarray,
    visit_count: np.ndarray,
    value_sum: np.ndarray,
) -> None:
    """Propagate value from leaf to root, flipping sign each level."""
    node = leaf
    while node >= 0:
        visit_count[node] += 1
        value_sum[node] += value
        value = -value
        node = parent[node]


@numba.njit(cache=True)
def _remove_vl_jit(
    path: np.ndarray,
    path_len: int,
    visit_count: np.ndarray,
    value_sum: np.ndarray,
    vl: float,
) -> None:
    """Remove virtual loss from nodes in path."""
    for i in range(path_len):
        node = path[i]
        visit_count[node] -= 1
        value_sum[node] += vl


# ── MCTSTree ──────────────────────────────────────────────────────────────


class MCTSTree:
    """Array-based MCTS tree with Numba-accelerated traversal.

    Children of a node occupy consecutive array indices for
    cache-friendly PUCT iteration.
    """

    __slots__ = [
        "capacity", "size", "root",
        "visit_count", "value_sum", "prior",
        "parent", "children_start", "num_children",
        "move_from_sq", "move_to_sq", "move_promo",
        "is_expanded", "is_terminal",
        "_path_buf",
    ]

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        self.capacity = capacity
        self.size: int = 0

        # float64 to match Python float precision for deterministic results
        self.visit_count = np.zeros(capacity, dtype=np.int32)
        self.value_sum = np.zeros(capacity, dtype=np.float64)
        self.prior = np.zeros(capacity, dtype=np.float64)

        self.parent = np.full(capacity, -1, dtype=np.int32)
        self.children_start = np.zeros(capacity, dtype=np.int32)
        self.num_children = np.zeros(capacity, dtype=np.int32)

        self.move_from_sq = np.zeros(capacity, dtype=np.int8)
        self.move_to_sq = np.zeros(capacity, dtype=np.int8)
        self.move_promo = np.zeros(capacity, dtype=np.int8)

        self.is_expanded = np.zeros(capacity, dtype=np.bool_)
        self.is_terminal = np.zeros(capacity, dtype=np.bool_)

        self._path_buf = np.zeros(_MAX_DEPTH, dtype=np.int32)

        self.root = self._alloc_node(-1)

    # ── Allocation ────────────────────────────────────────────────────

    def _alloc_node(self, parent_idx: int) -> int:
        idx = self.size
        if idx >= self.capacity:
            raise RuntimeError(f"MCTSTree exhausted ({self.capacity} nodes)")
        self.size += 1
        self.parent[idx] = parent_idx
        return idx

    def _alloc_children(self, parent_idx: int, count: int) -> int:
        start = self.size
        end = start + count
        if end > self.capacity:
            raise RuntimeError(
                f"MCTSTree exhausted ({self.capacity} nodes, need {count} more)"
            )
        self.size = end
        self.children_start[parent_idx] = start
        self.num_children[parent_idx] = count
        self.parent[start:end] = parent_idx
        return start

    def remaining_capacity(self) -> int:
        return self.capacity - self.size

    # ── Expand ────────────────────────────────────────────────────────

    def expand(self, node_idx: int, board: chess.Board, policy: np.ndarray) -> None:
        """Expand node with children for all legal moves."""
        self.is_expanded[node_idx] = True

        if board.is_game_over(claim_draw=False):
            self.is_terminal[node_idx] = True
            return

        moves: list[tuple[int, int, int, float]] = []
        total_prior = 0.0

        for move in board.legal_moves:
            try:
                idx = MoveEncoder.move_to_index(move)
                p = float(policy[idx])
                promo = move.promotion if move.promotion is not None else 0
                moves.append((move.from_square, move.to_square, promo, p))
                total_prior += p
            except ValueError:
                continue

        if not moves:
            self.is_terminal[node_idx] = True
            return

        nc = len(moves)
        start = self._alloc_children(node_idx, nc)

        # Match old MCTSNode: multiply by inverse for identical float results
        if total_prior > 0.0:
            inv_total = 1.0 / total_prior
            for i, (from_sq, to_sq, promo, p) in enumerate(moves):
                child = start + i
                self.move_from_sq[child] = from_sq
                self.move_to_sq[child] = to_sq
                self.move_promo[child] = promo
                self.prior[child] = p * inv_total
        else:
            uniform = 1.0 / nc
            for i, (from_sq, to_sq, promo, _p) in enumerate(moves):
                child = start + i
                self.move_from_sq[child] = from_sq
                self.move_to_sq[child] = to_sq
                self.move_promo[child] = promo
                self.prior[child] = uniform

    # ── Dirichlet noise ───────────────────────────────────────────────

    def add_dirichlet_noise(
        self, node_idx: int, alpha: float = 0.3, epsilon: float = 0.25,
    ) -> None:
        nc = self.num_children[node_idx]
        if nc == 0 or alpha <= 0.0 or epsilon <= 0.0:
            return
        start = self.children_start[node_idx]
        noise = np.random.dirichlet(np.full(nc, alpha, dtype=np.float64))
        # Per-element to match old MCTSNode float arithmetic exactly
        for i in range(nc):
            child = start + i
            self.prior[child] = (1 - epsilon) * self.prior[child] + epsilon * noise[i]

    # ── Traversal ─────────────────────────────────────────────────────

    def find_leaf_with_virtual_loss(
        self, board: chess.Board, cpuct: float, vl: float = 1.0,
    ) -> tuple[int, chess.Board, np.ndarray]:
        """Traverse to leaf via JIT, apply virtual loss.

        Returns (leaf_idx, leaf_board, path). path has node indices
        visited (excluding root). Board replay happens once at the end.
        """
        path_len = _find_leaf_jit(
            self.root,
            self.children_start, self.num_children,
            self.visit_count, self.value_sum, self.prior,
            self.is_expanded, self.is_terminal,
            cpuct, vl, self._path_buf,
        )
        path = self._path_buf[:path_len].copy()
        leaf = int(path[-1]) if path_len > 0 else self.root

        search_board = board.copy(stack=False)
        for i in range(path_len):
            node = int(path[i])
            from_sq = int(self.move_from_sq[node])
            to_sq = int(self.move_to_sq[node])
            promo = int(self.move_promo[node])
            move = chess.Move(from_sq, to_sq, promotion=promo if promo else None)
            search_board.push(move)

        return leaf, search_board, path

    def find_leaf(
        self, board: chess.Board, cpuct: float,
    ) -> tuple[int, chess.Board]:
        """Traverse to leaf WITHOUT virtual loss (single-threaded search)."""
        node = self.root
        search_board = board.copy(stack=False)
        while self.is_expanded[node] and not self.is_terminal[node]:
            child = int(_select_child_jit(
                node, self.children_start, self.num_children,
                self.visit_count, self.value_sum, self.prior, cpuct,
            ))
            search_board.push(self.get_child_move(child))
            node = child
        return node, search_board

    def remove_virtual_loss(self, path: np.ndarray, vl: float = 1.0) -> None:
        if len(path) > 0:
            _remove_vl_jit(path, len(path), self.visit_count, self.value_sum, vl)

    def backup(self, node_idx: int, value: float) -> None:
        _backup_jit(
            node_idx, value,
            self.parent, self.visit_count, self.value_sum,
        )

    # ── Query ─────────────────────────────────────────────────────────

    def get_child_move(self, child_idx: int) -> chess.Move:
        from_sq = int(self.move_from_sq[child_idx])
        to_sq = int(self.move_to_sq[child_idx])
        promo = int(self.move_promo[child_idx])
        return chess.Move(from_sq, to_sq, promotion=promo if promo else None)

    def get_visit_counts(self) -> dict[chess.Move, int]:
        nc = self.num_children[self.root]
        if nc == 0:
            return {}
        start = self.children_start[self.root]
        result: dict[chess.Move, int] = {}
        for i in range(nc):
            child = start + i
            result[self.get_child_move(child)] = int(self.visit_count[child])
        return result

    def get_child_for_move(self, move: chess.Move) -> Optional[int]:
        nc = self.num_children[self.root]
        start = self.children_start[self.root]
        from_sq = move.from_square
        to_sq = move.to_square
        promo = move.promotion if move.promotion is not None else 0
        for i in range(nc):
            child = start + i
            if (int(self.move_from_sq[child]) == from_sq
                    and int(self.move_to_sq[child]) == to_sq
                    and int(self.move_promo[child]) == promo):
                return child
        return None

    # ── Tree management ───────────────────────────────────────────────

    def reroot(self, child_idx: int) -> None:
        """Promote child to root. Old sibling nodes waste space but
        stay allocated until reset."""
        self.parent[child_idx] = -1
        self.root = child_idx

    def reset(self) -> None:
        """Reset to fresh single-root state."""
        s = self.size
        self.visit_count[:s] = 0
        self.value_sum[:s] = 0.0
        self.prior[:s] = 0.0
        self.parent[:s] = -1
        self.children_start[:s] = 0
        self.num_children[:s] = 0
        self.move_from_sq[:s] = 0
        self.move_to_sq[:s] = 0
        self.move_promo[:s] = 0
        self.is_expanded[:s] = False
        self.is_terminal[:s] = False
        self.size = 0
        self.root = self._alloc_node(-1)
