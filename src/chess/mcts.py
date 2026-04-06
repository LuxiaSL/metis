"""Monte Carlo Tree Search for chess.

Standard PUCT-based MCTS as used in AlphaZero:
1. SELECT: traverse tree using PUCT formula
2. EXPAND: add child nodes for the selected leaf
3. EVALUATE: neural network policy + value
4. BACKUP: propagate value up the tree

Supports batched evaluation for parallel games via BatchedMCTS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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
    temperature: float = 1.0
    temperature_threshold: int = 30  # Switch to greedy after this many ply


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
        """Mean action value Q(s, a)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def is_expanded(self) -> bool:
        return self._is_expanded

    def expand(
        self,
        board: chess.Board,
        policy: np.ndarray,
    ) -> None:
        """Expand this node with children for all legal moves.

        Args:
            board: Board state at this node.
            policy: Policy probabilities from neural network, shape (4672,).
        """
        self._is_expanded = True
        for move in board.legal_moves:
            try:
                idx = MoveEncoder.move_to_index(move)
                self.children[move] = MCTSNode(
                    parent=self, move=move, prior=policy[idx],
                )
            except ValueError:
                continue

    def select_child(self, cpuct: float) -> tuple[chess.Move, MCTSNode]:
        """Select child with highest PUCT score.

        PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
        """
        sqrt_parent = math.sqrt(self.visit_count)
        best_score = -float("inf")
        best_move: Optional[chess.Move] = None
        best_child: Optional[MCTSNode] = None

        for move, child in self.children.items():
            score = child.q_value + cpuct * child.prior * sqrt_parent / (1 + child.visit_count)
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        assert best_move is not None and best_child is not None
        return best_move, best_child

    def backup(self, value: float) -> None:
        """Propagate value up to root, flipping sign at each level."""
        node: Optional[MCTSNode] = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip for opponent's perspective
            node = node.parent


class MCTS:
    """Single-game MCTS with neural network evaluation."""

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
        """Evaluate a position with the neural network.

        Returns:
            (policy, value): policy is numpy (4672,), value is float.
        """
        encoded = BoardEncoder.encode_board(board).unsqueeze(0).to(self.device)

        self.model.eval()
        policy_logits, value = self.model(encoded)

        # Apply legal move mask and softmax to get policy probabilities
        mask = MoveEncoder.legal_move_mask(board).to(self.device)
        policy_logits = policy_logits.squeeze(0)
        policy_logits[~mask] = float("-inf")
        policy = torch.softmax(policy_logits, dim=0).cpu().numpy()

        return policy, value.item()

    def _terminal_value(self, board: chess.Board) -> float:
        """Get value for a terminal position from the perspective of the side to move."""
        result = board.result(claim_draw=True)
        if result == "1-0":
            return 1.0 if board.turn == chess.WHITE else -1.0
        elif result == "0-1":
            return 1.0 if board.turn == chess.BLACK else -1.0
        return 0.0  # Draw

    def search(self, board: chess.Board) -> dict[chess.Move, int]:
        """Run MCTS from the given position.

        Returns:
            Visit counts for each child of the root node.
        """
        if board.is_game_over(claim_draw=True):
            return {}

        # Create and expand root
        root = MCTSNode()
        policy, _ = self._evaluate(board)

        # Add Dirichlet noise at root for exploration
        legal_moves = list(board.legal_moves)
        noise = np.random.dirichlet(
            [self.config.dirichlet_alpha] * len(legal_moves),
        )
        eps = self.config.dirichlet_epsilon
        noisy_policy = policy.copy()
        for i, move in enumerate(legal_moves):
            try:
                idx = MoveEncoder.move_to_index(move)
                noisy_policy[idx] = (1 - eps) * policy[idx] + eps * noise[i]
            except ValueError:
                continue

        root.expand(board, noisy_policy)

        # Run simulations
        for _ in range(self.config.num_simulations):
            node = root
            search_board = board.copy()

            # SELECT: traverse tree
            while node.is_expanded and not search_board.is_game_over(claim_draw=True):
                move, node = node.select_child(self.config.cpuct)
                search_board.push(move)

            # EVALUATE
            if search_board.is_game_over(claim_draw=True):
                value = self._terminal_value(search_board)
            else:
                # EXPAND + EVALUATE
                policy, value = self._evaluate(search_board)
                node.expand(search_board, policy)

            # BACKUP (value is from the perspective of the node's side to move)
            node.backup(value)

        return {move: child.visit_count for move, child in root.children.items()}

    def select_move(
        self,
        visit_counts: dict[chess.Move, int],
        temperature: float = 1.0,
    ) -> chess.Move:
        """Select a move from visit counts with temperature scaling.

        Args:
            visit_counts: Move → visit count from MCTS search.
            temperature: 0 = greedy (pick most visited), >0 = proportional.

        Returns:
            Selected chess.Move.
        """
        moves = list(visit_counts.keys())
        counts = np.array([visit_counts[m] for m in moves], dtype=np.float64)

        if temperature == 0 or len(moves) == 1:
            return moves[int(np.argmax(counts))]

        # Temperature-scaled sampling
        counts = counts ** (1.0 / temperature)
        probs = counts / counts.sum()
        idx = np.random.choice(len(moves), p=probs)
        return moves[idx]


class BatchedMCTS:
    """MCTS for multiple parallel games with batched neural network evaluation.

    Instead of evaluating one position at a time, collects pending evaluations
    from multiple games and batches them into a single GPU forward pass.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: MCTSConfig,
        device: torch.device,
        num_parallel: int = 64,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.num_parallel = num_parallel

    @torch.no_grad()
    def _batched_evaluate(
        self, boards: list[chess.Board],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate multiple positions in one forward pass.

        Returns:
            policies: (N, 4672) numpy array
            values: (N,) numpy array
        """
        if not boards:
            return np.empty((0, POLICY_SIZE)), np.empty(0)

        encoded = BoardEncoder.encode_board_batch(boards).to(self.device)

        self.model.eval()
        policy_logits, values = self.model(encoded)

        # Apply legal move masks and softmax
        policies = np.zeros((len(boards), POLICY_SIZE), dtype=np.float32)
        for i, board in enumerate(boards):
            mask = MoveEncoder.legal_move_mask(board).to(self.device)
            logits = policy_logits[i].clone()
            logits[~mask] = float("-inf")
            policies[i] = torch.softmax(logits, dim=0).cpu().numpy()

        return policies, values.squeeze(-1).cpu().numpy()

    def search_batch(
        self,
        boards: list[chess.Board],
    ) -> list[dict[chess.Move, int]]:
        """Run MCTS for multiple positions simultaneously.

        Uses batched NN evaluation: each simulation step collects
        all pending leaf evaluations and processes them in one GPU call.

        Args:
            boards: List of board positions to search.

        Returns:
            List of visit count dicts, one per board.
        """
        n = len(boards)
        roots: list[MCTSNode] = [MCTSNode() for _ in range(n)]

        # Initial expansion: batch-evaluate all root positions
        active_boards = [b for b in boards if not b.is_game_over(claim_draw=True)]
        active_indices = [i for i, b in enumerate(boards) if not b.is_game_over(claim_draw=True)]

        if active_boards:
            policies, _ = self._batched_evaluate(active_boards)

            for j, idx in enumerate(active_indices):
                policy = policies[j]
                legal_moves = list(boards[idx].legal_moves)

                # Add Dirichlet noise at root
                noise = np.random.dirichlet(
                    [self.config.dirichlet_alpha] * len(legal_moves),
                )
                eps = self.config.dirichlet_epsilon
                noisy_policy = policy.copy()
                for k, move in enumerate(legal_moves):
                    try:
                        mi = MoveEncoder.move_to_index(move)
                        noisy_policy[mi] = (1 - eps) * policy[mi] + eps * noise[k]
                    except ValueError:
                        continue

                roots[idx].expand(boards[idx], noisy_policy)

        # Run simulations
        for _ in range(self.config.num_simulations):
            # For each game, traverse to a leaf
            leaves: list[tuple[int, MCTSNode, chess.Board]] = []

            for i in active_indices:
                node = roots[i]
                search_board = boards[i].copy()

                while node.is_expanded and not search_board.is_game_over(claim_draw=True):
                    move, node = node.select_child(self.config.cpuct)
                    search_board.push(move)

                if search_board.is_game_over(claim_draw=True):
                    # Terminal: backup immediately
                    result = search_board.result(claim_draw=True)
                    if result == "1-0":
                        value = 1.0 if search_board.turn == chess.WHITE else -1.0
                    elif result == "0-1":
                        value = 1.0 if search_board.turn == chess.BLACK else -1.0
                    else:
                        value = 0.0
                    node.backup(value)
                else:
                    leaves.append((i, node, search_board))

            # Batch evaluate all non-terminal leaves
            if leaves:
                leaf_boards = [lb for _, _, lb in leaves]
                policies, values = self._batched_evaluate(leaf_boards)

                for j, (game_idx, node, lb) in enumerate(leaves):
                    node.expand(lb, policies[j])
                    node.backup(values[j])

        return [
            {move: child.visit_count for move, child in root.children.items()}
            for root in roots
        ]
