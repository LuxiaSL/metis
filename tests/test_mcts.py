import unittest

import chess
import numpy as np
import torch

from src.chess.board import POLICY_SIZE
from src.chess.mcts import BatchedMCTS, MCTS, MCTSConfig, MCTSNode


class DummyModel(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        policy = torch.zeros((batch_size, POLICY_SIZE), device=x.device)
        value = torch.zeros((batch_size, 1), device=x.device)
        return policy, value


class MCTSTest(unittest.TestCase):
    def test_expand_renormalizes_legal_priors(self) -> None:
        board = chess.Board()
        policy = np.full(POLICY_SIZE, 1.0 / POLICY_SIZE, dtype=np.float32)

        root = MCTSNode()
        root.expand(board, policy)

        priors = [child.prior for child in root.children.values()]
        self.assertEqual(len(priors), len(list(board.legal_moves)))
        self.assertAlmostEqual(sum(priors), 1.0, places=6)

        expected = 1.0 / len(priors)
        for prior in priors:
            self.assertAlmostEqual(prior, expected, places=6)

    def test_batched_search_honors_num_simulations(self) -> None:
        device = torch.device("cpu")
        board = chess.Board()

        for sims, nvl in [(3, 4), (5, 4), (8, 4), (9, 4)]:
            mcts = BatchedMCTS(
                DummyModel(),
                MCTSConfig(
                    num_simulations=sims,
                    num_virtual_leaves=nvl,
                    dirichlet_alpha=0.0,
                    dirichlet_epsilon=0.0,
                ),
                device,
            )
            visit_counts = mcts.search_batch([board])[0]
            self.assertEqual(sum(visit_counts.values()), sims)

    def test_search_allows_zero_dirichlet_noise(self) -> None:
        device = torch.device("cpu")
        mcts = MCTS(
            DummyModel(),
            MCTSConfig(
                num_simulations=1,
                dirichlet_alpha=0.0,
                dirichlet_epsilon=0.0,
            ),
            device,
        )

        visit_counts = mcts.search(chess.Board())
        self.assertEqual(sum(visit_counts.values()), 1)


if __name__ == "__main__":
    unittest.main()
