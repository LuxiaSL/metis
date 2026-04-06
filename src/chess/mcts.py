"""Monte Carlo Tree Search for chess.

Standard PUCT-based MCTS as used in AlphaZero:
1. SELECT: traverse tree using PUCT formula
2. EXPAND: add child nodes for the selected leaf
3. EVALUATE: neural network policy + value
4. BACKUP: propagate value up the tree

Supports:
- Virtual loss for parallel leaf selection within a game
- Tree reuse between moves
- Batched evaluation for parallel games (BatchedMCTS)
- Multiprocessing self-play with centralized GPU evaluation (ParallelSelfPlay)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import chess
import numpy as np
import torch

from src.chess.board import BoardEncoder, MoveEncoder, POLICY_SIZE


@dataclass
class MCTSConfig:
    """MCTS hyperparameters."""

    num_simulations: int = 800
    cpuct: float = 1.25
    dirichlet_alpha: float = 0.3   # Chess = 0.3 (AlphaZero)
    dirichlet_epsilon: float = 0.25
    num_virtual_leaves: int = 4    # Parallel leaves per game via virtual loss


class MCTSNode:
    """Single node in the MCTS tree."""

    __slots__ = [
        "parent", "move", "children", "visit_count",
        "value_sum", "prior", "_is_expanded",
    ]

    def __init__(
        self,
        parent: Optional[MCTSNode] = None,
        move: Optional[chess.Move] = None,
        prior: float = 0.0,
    ) -> None:
        self.parent = parent
        self.move = move
        self.children: dict[chess.Move, MCTSNode] = {}
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.prior: float = prior
        self._is_expanded: bool = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def is_expanded(self) -> bool:
        return self._is_expanded

    def expand(self, board: chess.Board, policy: np.ndarray) -> None:
        """Expand with children for all legal moves."""
        self._is_expanded = True
        children: list[MCTSNode] = []
        total_prior = 0.0

        for move in board.legal_moves:
            try:
                idx = MoveEncoder.move_to_index(move)
                child = MCTSNode(parent=self, move=move, prior=float(policy[idx]))
                self.children[move] = child
                children.append(child)
                total_prior += child.prior
            except ValueError:
                continue

        if not children:
            return

        if total_prior > 0.0:
            inv_total = 1.0 / total_prior
            for child in children:
                child.prior *= inv_total
        else:
            uniform_prior = 1.0 / len(children)
            for child in children:
                child.prior = uniform_prior

    def select_child(self, cpuct: float) -> tuple[chess.Move, MCTSNode]:
        """Select child with highest PUCT score."""
        exploration_scale = cpuct * math.sqrt(max(self.visit_count, 1))
        best_score = -float("inf")
        best_child: Optional[MCTSNode] = None

        for child in self.children.values():
            visits = child.visit_count
            q_value = child.value_sum / visits if visits else 0.0
            score = q_value + exploration_scale * child.prior / (1 + visits)
            if score > best_score:
                best_score = score
                best_child = child

        assert best_child is not None and best_child.move is not None
        return best_child.move, best_child

    def backup(self, value: float) -> None:
        """Propagate value up to root, flipping sign at each level."""
        node: Optional[MCTSNode] = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value
            node = node.parent

    def make_root(self) -> None:
        """Promote this node to root (for tree reuse between moves)."""
        self.parent = None

    def add_dirichlet_noise(
        self, alpha: float = 0.3, epsilon: float = 0.25,
    ) -> None:
        """Add Dirichlet noise to children priors (root exploration)."""
        children = list(self.children.values())
        if not children or alpha <= 0.0 or epsilon <= 0.0:
            return
        noise = np.random.dirichlet(np.full(len(children), alpha, dtype=np.float64))
        for i, child in enumerate(children):
            child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]


# ── Virtual loss helpers ───────────────────────────────────────────────────

def find_leaf_with_virtual_loss(
    root: MCTSNode,
    board: chess.Board,
    cpuct: float,
    vl: float = 1.0,
) -> tuple[MCTSNode, chess.Board, list[MCTSNode]]:
    """Traverse tree to a leaf, applying virtual loss along the path.

    Virtual loss discourages other parallel searches from the same path.

    Args:
        root: Root node.
        board: Board at root position.
        cpuct: PUCT exploration constant.
        vl: Virtual loss magnitude.

    Returns:
        (leaf_node, leaf_board, path): path is list of nodes visited
            (excluding root, including leaf).
    """
    path: list[MCTSNode] = []
    node = root
    search_board = board.copy(stack=False)

    while node.is_expanded and not search_board.is_game_over(claim_draw=False):
        move, child = node.select_child(cpuct)
        # Apply virtual loss: inflate visits, add pessimistic value
        child.visit_count += 1
        child.value_sum -= vl
        path.append(child)
        node = child
        search_board.push(move)

    return node, search_board, path


def remove_virtual_loss(path: list[MCTSNode], vl: float = 1.0) -> None:
    """Remove virtual loss from all nodes in the path."""
    for node in path:
        node.visit_count -= 1
        node.value_sum += vl


def terminal_value(board: chess.Board) -> float:
    """Value for an automatic terminal position from the side to move's perspective."""
    result = board.result(claim_draw=False)
    if result == "1-0":
        return 1.0 if board.turn == chess.WHITE else -1.0
    elif result == "0-1":
        return 1.0 if board.turn == chess.BLACK else -1.0
    return 0.0


def select_move(
    visit_counts: dict[chess.Move, int],
    temperature: float = 1.0,
) -> chess.Move:
    """Select a move from visit counts with temperature scaling."""
    moves = list(visit_counts.keys())
    counts = np.array([visit_counts[m] for m in moves], dtype=np.float64)

    if temperature == 0 or len(moves) == 1:
        return moves[int(np.argmax(counts))]

    total = counts.sum()
    if total <= 0:
        return moves[int(np.random.choice(len(moves)))]

    counts = counts ** (1.0 / temperature)
    probs = counts / counts.sum()
    return moves[np.random.choice(len(moves), p=probs)]


# ── Single-game MCTS (for evaluation) ─────────────────────────────────────


class MCTS:
    """Single-game MCTS with neural network evaluation. Used for Stockfish eval."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: MCTSConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device

    @torch.no_grad()
    def _evaluate(self, board: chess.Board) -> tuple[np.ndarray, float]:
        """Evaluate one position with the neural network."""
        encoded = BoardEncoder.encode_board(board).unsqueeze(0).to(self.device)
        self.model.eval()
        policy_logits, value = self.model(encoded)

        policy = torch.softmax(policy_logits.squeeze(0).float(), dim=0).cpu().numpy()
        return policy, value.item()

    def search(self, board: chess.Board) -> dict[chess.Move, int]:
        """Run MCTS. Returns move visit counts."""
        if board.is_game_over(claim_draw=True):
            return {}

        root = MCTSNode()
        policy, _ = self._evaluate(board)
        root.expand(board, policy)
        root.add_dirichlet_noise(self.config.dirichlet_alpha, self.config.dirichlet_epsilon)

        for _ in range(self.config.num_simulations):
            node = root
            search_board = board.copy(stack=False)

            while node.is_expanded and not search_board.is_game_over(claim_draw=False):
                move, node = node.select_child(self.config.cpuct)
                search_board.push(move)

            if search_board.is_game_over(claim_draw=False):
                value = terminal_value(search_board)
            else:
                policy, value = self._evaluate(search_board)
                node.expand(search_board, policy)

            node.backup(value)

        return {move: child.visit_count for move, child in root.children.items()}


# ── Batched MCTS (single-process, for fallback) ───────────────────────────


class BatchedMCTS:
    """MCTS for multiple parallel games with batched NN evaluation.

    Single-process fallback for when multiprocessing isn't needed (eval, testing).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: MCTSConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device

    @torch.no_grad()
    def _batched_evaluate(
        self, boards: list[chess.Board],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate multiple positions in one forward pass."""
        if not boards:
            return np.empty((0, POLICY_SIZE)), np.empty(0)

        encoded = BoardEncoder.encode_board_batch(boards).to(self.device)
        self.model.eval()
        policy_logits, values = self.model(encoded)

        policies = torch.softmax(policy_logits.float(), dim=-1).cpu().numpy()
        return policies, values.squeeze(-1).float().cpu().numpy()

    def search_batch(
        self, boards: list[chess.Board],
    ) -> list[dict[chess.Move, int]]:
        """Run MCTS for multiple positions with batched evaluation + virtual loss."""
        n = len(boards)
        roots: list[MCTSNode] = [MCTSNode() for _ in range(n)]
        active = [i for i, b in enumerate(boards) if not b.is_game_over(claim_draw=True)]

        # Initial expansion
        if active:
            active_boards = [boards[i] for i in active]
            policies, _ = self._batched_evaluate(active_boards)
            for j, idx in enumerate(active):
                roots[idx].expand(boards[idx], policies[j])
                roots[idx].add_dirichlet_noise(
                    self.config.dirichlet_alpha, self.config.dirichlet_epsilon,
                )

        # Simulations with virtual loss batching
        nvl = self.config.num_virtual_leaves

        for sim_start in range(0, self.config.num_simulations, nvl):
            leaves_per_game = min(nvl, self.config.num_simulations - sim_start)
            leaves: list[tuple[int, MCTSNode, chess.Board, list[MCTSNode]]] = []

            for i in active:
                for _ in range(leaves_per_game):
                    leaf, search_board, path = find_leaf_with_virtual_loss(
                        roots[i], boards[i], self.config.cpuct,
                    )
                    if search_board.is_game_over(claim_draw=False):
                        remove_virtual_loss(path)
                        leaf.backup(terminal_value(search_board))
                    else:
                        leaves.append((i, leaf, search_board, path))

            if leaves:
                leaf_boards = [lb for _, _, lb, _ in leaves]
                policies, values = self._batched_evaluate(leaf_boards)

                for j, (_, leaf, _, path) in enumerate(leaves):
                    leaf.expand(leaf_boards[j], policies[j])
                    remove_virtual_loss(path)
                    leaf.backup(values[j])

        return [
            {move: child.visit_count for move, child in root.children.items()}
            for root in roots
        ]
