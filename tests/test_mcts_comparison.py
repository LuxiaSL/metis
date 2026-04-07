"""Comparison test: old MCTSNode vs new MCTSTree produce identical results."""

import unittest

import chess
import numpy as np

from src.chess.board import POLICY_SIZE
from src.chess.mcts import (
    MCTSNode, find_leaf_with_virtual_loss, remove_virtual_loss, terminal_value,
)
from src.chess.mcts_array import MCTSTree


class TestMCTSComparison(unittest.TestCase):
    """Verify MCTSTree matches MCTSNode exactly given same inputs."""

    def _make_policy(self, seed: int = 42) -> np.ndarray:
        return np.random.RandomState(seed).rand(POLICY_SIZE).astype(np.float64)

    def test_expand_priors_match(self) -> None:
        """Expansion produces identical priors."""
        board = chess.Board()
        policy = self._make_policy()

        old = MCTSNode()
        old.expand(board, policy)

        tree = MCTSTree(capacity=10_000)
        tree.expand(tree.root, board, policy)

        nc = tree.num_children[tree.root]
        start = tree.children_start[tree.root]

        old_children = list(old.children.values())
        self.assertEqual(len(old_children), nc)

        for i, old_child in enumerate(old_children):
            child = start + i
            self.assertAlmostEqual(
                old_child.prior, tree.prior[child], places=14,
                msg=f"Prior mismatch at child {i}",
            )
            self.assertEqual(old_child.move.from_square, int(tree.move_from_sq[child]))
            self.assertEqual(old_child.move.to_square, int(tree.move_to_sq[child]))

    def test_dirichlet_noise_match(self) -> None:
        """Dirichlet noise produces identical priors."""
        board = chess.Board()
        policy = self._make_policy()

        np.random.seed(999)
        old = MCTSNode()
        old.expand(board, policy)
        old.add_dirichlet_noise(0.3, 0.25)

        np.random.seed(999)
        tree = MCTSTree(capacity=10_000)
        tree.expand(tree.root, board, policy)
        tree.add_dirichlet_noise(tree.root, 0.3, 0.25)

        old_priors = [c.prior for c in old.children.values()]
        start = tree.children_start[tree.root]
        nc = tree.num_children[tree.root]
        new_priors = [tree.prior[start + i] for i in range(nc)]

        for i, (op, np_) in enumerate(zip(old_priors, new_priors)):
            self.assertAlmostEqual(op, np_, places=14, msg=f"Noise mismatch at {i}")

    def test_no_vl_search_identical(self) -> None:
        """Single-threaded search (no VL) produces identical visit counts."""
        board = chess.Board()
        policy = self._make_policy()
        cpuct = 1.25
        num_sims = 100

        # Old
        np.random.seed(123)
        old_root = MCTSNode()
        old_root.expand(board, policy)
        old_root.add_dirichlet_noise(0.3, 0.25)

        for _ in range(num_sims):
            node = old_root
            sb = board.copy(stack=False)
            while node.is_expanded and not node._is_terminal:
                move, node = node.select_child(cpuct)
                sb.push(move)
            if node._is_terminal or sb.is_game_over(claim_draw=False):
                value = terminal_value(sb)
            else:
                node.expand(sb, policy)
                value = 0.0
            node.backup(value)
        old_vc = {m: c.visit_count for m, c in old_root.children.items()}

        # New
        np.random.seed(123)
        tree = MCTSTree(capacity=50_000)
        tree.expand(tree.root, board, policy)
        tree.add_dirichlet_noise(tree.root, 0.3, 0.25)

        for _ in range(num_sims):
            node, sb = tree.find_leaf(board, cpuct)
            if tree.is_terminal[node] or sb.is_game_over(claim_draw=False):
                value = terminal_value(sb)
            else:
                tree.expand(node, sb, policy)
                value = 0.0
            tree.backup(node, value)
        new_vc = tree.get_visit_counts()

        self.assertEqual(set(old_vc.keys()), set(new_vc.keys()))
        for move in old_vc:
            self.assertEqual(old_vc[move], new_vc[move], f"Mismatch for {move}")

    def test_virtual_loss_search_identical(self) -> None:
        """Virtual-loss batched search produces identical visit counts."""
        board = chess.Board()
        policy = self._make_policy()
        cpuct = 1.25
        num_sims = 80
        nvl = 4

        # Old
        np.random.seed(456)
        old_root = MCTSNode()
        old_root.expand(board, policy)
        old_root.add_dirichlet_noise(0.3, 0.25)

        for sim_start in range(0, num_sims, nvl):
            lpg = min(nvl, num_sims - sim_start)
            leaves = []
            for _ in range(lpg):
                leaf, sb, path = find_leaf_with_virtual_loss(old_root, board, cpuct)
                if leaf._is_terminal or sb.is_game_over(claim_draw=False):
                    remove_virtual_loss(path)
                    leaf.backup(terminal_value(sb))
                else:
                    leaves.append((leaf, sb, path))
            for leaf, sb, path in leaves:
                leaf.expand(sb, policy)
                remove_virtual_loss(path)
                leaf.backup(0.0)

        old_vc = {m: c.visit_count for m, c in old_root.children.items()}

        # New
        np.random.seed(456)
        tree = MCTSTree(capacity=50_000)
        tree.expand(tree.root, board, policy)
        tree.add_dirichlet_noise(tree.root, 0.3, 0.25)

        for sim_start in range(0, num_sims, nvl):
            lpg = min(nvl, num_sims - sim_start)
            leaves = []
            for _ in range(lpg):
                leaf, sb, path = tree.find_leaf_with_virtual_loss(board, cpuct)
                if tree.is_terminal[leaf] or sb.is_game_over(claim_draw=False):
                    tree.remove_virtual_loss(path)
                    tree.backup(leaf, terminal_value(sb))
                else:
                    leaves.append((leaf, sb, path))
            for leaf, sb, path in leaves:
                tree.expand(leaf, sb, policy)
                tree.remove_virtual_loss(path)
                tree.backup(leaf, 0.0)

        new_vc = tree.get_visit_counts()

        self.assertEqual(set(old_vc.keys()), set(new_vc.keys()))
        for move in old_vc:
            self.assertEqual(old_vc[move], new_vc[move], f"Mismatch for {move}")

    def test_midgame_position(self) -> None:
        """Works on a non-starting position (Sicilian)."""
        board = chess.Board()
        for uci in ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4"]:
            board.push_uci(uci)

        policy = self._make_policy(seed=77)
        cpuct = 1.25
        num_sims = 60

        np.random.seed(777)
        old_root = MCTSNode()
        old_root.expand(board, policy)
        old_root.add_dirichlet_noise(0.3, 0.25)
        for _ in range(num_sims):
            node = old_root
            sb = board.copy(stack=False)
            while node.is_expanded and not node._is_terminal:
                move, node = node.select_child(cpuct)
                sb.push(move)
            if node._is_terminal or sb.is_game_over(claim_draw=False):
                node.backup(terminal_value(sb))
            else:
                node.expand(sb, policy)
                node.backup(0.0)
        old_vc = {m: c.visit_count for m, c in old_root.children.items()}

        np.random.seed(777)
        tree = MCTSTree(capacity=50_000)
        tree.expand(tree.root, board, policy)
        tree.add_dirichlet_noise(tree.root, 0.3, 0.25)
        for _ in range(num_sims):
            node, sb = tree.find_leaf(board, cpuct)
            if tree.is_terminal[node] or sb.is_game_over(claim_draw=False):
                tree.backup(node, terminal_value(sb))
            else:
                tree.expand(node, sb, policy)
                tree.backup(node, 0.0)
        new_vc = tree.get_visit_counts()

        self.assertEqual(set(old_vc.keys()), set(new_vc.keys()))
        for move in old_vc:
            self.assertEqual(old_vc[move], new_vc[move], f"Mismatch for {move}")


if __name__ == "__main__":
    unittest.main()
