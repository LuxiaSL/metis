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

from src.chess.board import BoardEncoder, MoveEncoder, POLICY_SIZE, mirror_policy
from src.chess.mcts_array import MCTSTree


def _extract_policy_and_value(outputs) -> tuple[torch.Tensor, torch.Tensor]:
    """Accept both legacy (policy, value) and current multi-head outputs."""
    if not isinstance(outputs, tuple) or len(outputs) < 2:
        raise ValueError("Model must return at least (policy, value_or_wdl)")

    policy_logits = outputs[0]
    value_head = outputs[1]
    if value_head.shape[-1] == 3:
        wdl_probs = torch.softmax(value_head.float(), dim=-1)
        values = wdl_probs[..., 2] - wdl_probs[..., 0]
    else:
        values = value_head.squeeze(-1).float()
    return policy_logits, values


@dataclass
class MCTSConfig:
    """MCTS hyperparameters."""

    num_simulations: int = 800
    cpuct: float = 1.25
    dirichlet_alpha: float = 0.3   # Chess = 0.3 (AlphaZero)
    dirichlet_epsilon: float = 0.25
    num_virtual_leaves: int = 8    # Parallel leaves per game via virtual loss


class MCTSNode:
    """Single node in the MCTS tree."""

    __slots__ = [
        "parent", "move", "children", "visit_count",
        "value_sum", "prior", "_is_expanded", "_is_terminal",
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
        self._is_terminal: bool = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def is_expanded(self) -> bool:
        return self._is_expanded

    def expand(self, board: chess.Board, policy: np.ndarray) -> None:
        """Expand with children for all legal moves.

        Also checks for terminal positions (insufficient material, 75-move rule,
        fivefold repetition) so traversal can skip is_game_over() at every level.
        """
        self._is_expanded = True

        # Check terminal status once during expansion (saves ~5μs × depth × sims
        # by avoiding redundant is_game_over calls during tree traversal)
        if board.is_game_over(claim_draw=False):
            self._is_terminal = True
            return

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
            self._is_terminal = True
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

    while node.is_expanded and not node._is_terminal:
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


# ── Gumbel AlphaZero ──────────────────────────────────────────────────────


@dataclass
class GumbelConfig:
    """Gumbel AlphaZero hyperparameters."""

    num_simulations: int = 200
    max_K: int = 16
    c_visit: float = 50.0
    cpuct: float = 1.25
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.0  # Disabled — Gumbel provides exploration


def gumbel_top_k(
    policy_logits: np.ndarray,
    legal_move_indices: list[int],
    K: int,
) -> tuple[list[int], np.ndarray]:
    """Sample K actions without replacement using the Gumbel-Top-K trick.

    Args:
        policy_logits: Raw logits from network (4672,).
        legal_move_indices: Indices into policy_logits for legal moves.
        K: Number of candidate actions to select.

    Returns:
        selected_indices: K indices into the policy_logits array (not into legal_move_indices).
        gumbel_scores: The (log_pi + gumbel) scores for the selected actions.
    """
    log_pi = policy_logits[legal_move_indices].astype(np.float64)
    log_pi = log_pi - np.max(log_pi)  # Numerical stability
    gumbels = -np.log(-np.log(np.random.uniform(size=len(log_pi)) + 1e-20) + 1e-20)
    scores = log_pi + gumbels

    K = min(K, len(scores))
    if K >= len(scores):
        top_K_local = np.argsort(scores)[::-1]
    else:
        top_K_local = np.argpartition(scores, -K)[-K:]
        top_K_local = top_K_local[np.argsort(scores[top_K_local])[::-1]]

    selected = [legal_move_indices[i] for i in top_K_local]
    return selected, scores[top_K_local]


def compute_sigma(completed_q: dict[chess.Move, float], c_visit: float) -> float:
    """Compute sigma for the improved policy from completed Q-value range.

    sigma = c_visit * (max_Q - min_Q + epsilon)
    """
    if not completed_q:
        return c_visit * 0.01
    q_vals = list(completed_q.values())
    q_range = max(q_vals) - min(q_vals)
    return c_visit * (q_range + 0.01)


def compute_improved_policy(
    policy_logits: np.ndarray,
    completed_q: dict[chess.Move, float],
    legal_moves: list[chess.Move],
    sigma: float,
) -> np.ndarray:
    """Compute improved policy: softmax(logits + sigma * Q_completed).

    Args:
        policy_logits: Raw network logits (4672,).
        completed_q: Q-value for every legal move.
        legal_moves: All legal moves at root.
        sigma: Scaling for Q-values.

    Returns:
        (4672,) policy distribution.
    """
    policy = np.zeros(POLICY_SIZE, dtype=np.float32)
    if not legal_moves:
        return policy

    indices = [MoveEncoder.move_to_index(m) for m in legal_moves]
    logits = policy_logits[indices].astype(np.float64)
    q_vals = np.array([completed_q.get(m, 0.0) for m in legal_moves], dtype=np.float64)

    improved = logits + sigma * q_vals
    improved -= improved.max()
    exp_improved = np.exp(improved)
    probs = exp_improved / (exp_improved.sum() + 1e-8)

    for i, idx in enumerate(indices):
        policy[idx] = float(probs[i])

    return policy


def sequential_halving(
    tree: MCTSTree,
    board: chess.Board,
    candidate_moves: list[chess.Move],
    num_simulations: int,
    eval_fn,
    cpuct: float = 1.25,
    num_virtual_leaves: int = 1,
) -> chess.Move:
    """Allocate simulations via Sequential Halving, return winner.

    Uses virtual leaves to batch multiple leaf evaluations per eval call,
    reducing round-trip overhead while keeping per-game independence
    (no cross-game synchronization).

    Args:
        tree: MCTS tree with root already expanded.
        board: Board at root position.
        candidate_moves: K moves from Gumbel-Top-K.
        num_simulations: Total simulation budget.
        eval_fn: Callable(board) -> (policy, value) when nvl=1, or
                 Callable(list[board]) -> (policies, values) when nvl>1.
        cpuct: PUCT constant for non-root traversal.
        num_virtual_leaves: Leaves to find per eval call (1 = original behavior).

    Returns:
        The winning move after Sequential Halving.
    """
    remaining = list(candidate_moves)
    num_phases = max(1, int(np.ceil(np.log2(len(remaining)))))
    sims_per_phase = max(1, num_simulations // num_phases)
    nvl = num_virtual_leaves

    for phase in range(num_phases):
        if len(remaining) <= 1:
            break

        sims_per_action = max(1, sims_per_phase // len(remaining))

        for move in remaining:
            child_idx = tree.get_child_for_move(move)
            if child_idx is None:
                continue

            child_board = board.copy(stack=False)
            child_board.push(move)

            sims_done = 0
            while sims_done < sims_per_action:
                # Handle terminal / unexpanded child (no batching needed)
                if tree.is_terminal[child_idx]:
                    tree.backup(child_idx, terminal_value(child_board))
                    sims_done += 1
                    continue

                if not tree.is_expanded[child_idx]:
                    if nvl > 1:
                        policies, values = eval_fn([child_board])
                        tree.expand(child_idx, child_board, policies[0])
                        tree.backup(child_idx, float(values[0]))
                    else:
                        policy, value = eval_fn(child_board)
                        tree.expand(child_idx, child_board, policy)
                        tree.backup(child_idx, value)
                    sims_done += 1
                    continue

                # Find up to nvl leaves with virtual loss
                batch_size = min(nvl, sims_per_action - sims_done)
                leaves: list[tuple[int, object, np.ndarray]] = []  # (leaf_idx, leaf_board, path)

                for _ in range(batch_size):
                    leaf_idx, leaf_board, path = tree.find_leaf_in_subtree_vl(
                        child_idx, child_board, cpuct,
                    )
                    if tree.is_terminal[leaf_idx] or leaf_board.is_game_over(claim_draw=False):
                        tree.remove_virtual_loss(path)
                        tree.backup(leaf_idx, terminal_value(leaf_board))
                        sims_done += 1
                    else:
                        leaves.append((leaf_idx, leaf_board, path))

                if not leaves:
                    continue

                # Batch evaluate all non-terminal leaves
                if nvl > 1:
                    leaf_boards = [lb for _, lb, _ in leaves]
                    policies, values = eval_fn(leaf_boards)
                    for k, (leaf_idx, lb, path) in enumerate(leaves):
                        tree.expand(leaf_idx, lb, policies[k])
                        tree.remove_virtual_loss(path)
                        tree.backup(leaf_idx, float(values[k]))
                else:
                    # Single leaf (nvl=1 path)
                    leaf_idx, lb, path = leaves[0]
                    policy, value = eval_fn(lb)
                    tree.expand(leaf_idx, lb, policy)
                    tree.remove_virtual_loss(path)
                    tree.backup(leaf_idx, value)

                sims_done += len(leaves)

        q_values = tree.get_child_q_values(negate=True)
        remaining.sort(key=lambda m: q_values.get(m, float('-inf')), reverse=True)
        remaining = remaining[:max(1, len(remaining) // 2)]

    return remaining[0]


@dataclass
class _GumbelGameState:
    """Per-game state tracked during batched Sequential Halving."""
    game_idx: int
    tree: MCTSTree
    board: chess.Board  # Root board
    remaining_moves: list[chess.Move]
    child_boards: dict  # move → board with move applied (cached)
    sims_done: dict  # move → sims completed this phase


def sequential_halving_batched(
    game_states: list[tuple[int, MCTSTree, chess.Board, list[chess.Move]]],
    num_simulations: int,
    eval_batch_fn,
    cpuct: float = 1.25,
    num_virtual_leaves: int = 8,
) -> dict[int, chess.Move]:
    """Batched Sequential Halving across multiple games with virtual leaves.

    Collects leaves from ALL games × ALL remaining actions × virtual leaves,
    evaluates them in one batched NN call, then expands and backs up.

    Args:
        game_states: List of (game_idx, tree, board, candidate_moves) tuples.
        num_simulations: Total simulation budget per game.
        eval_batch_fn: Callable(list[Board]) -> (policies_batch, values_batch).
                       Evaluates multiple boards in one GPU batch.
        cpuct: PUCT constant for non-root traversal.
        num_virtual_leaves: Max parallel leaves per action per sim round.

    Returns:
        Dict mapping game_idx → winning move.
    """
    if not game_states:
        return {}

    # Initialize per-game state
    games: list[_GumbelGameState] = []
    for game_idx, tree, board, candidates in game_states:
        child_boards: dict = {}
        for move in candidates:
            cb = board.copy(stack=False)
            cb.push(move)
            child_boards[move] = cb
        games.append(_GumbelGameState(
            game_idx=game_idx,
            tree=tree,
            board=board,
            remaining_moves=list(candidates),
            child_boards=child_boards,
            sims_done={m: 0 for m in candidates},
        ))

    # Compute phase structure (use max K across games for uniformity)
    max_K = max(len(g.remaining_moves) for g in games) if games else 1
    num_phases = max(1, int(np.ceil(np.log2(max_K))))
    sims_per_phase = max(1, num_simulations // num_phases)

    for phase in range(num_phases):
        # Compute sims_per_action for each game (may differ as K shrinks)
        for g in games:
            n_remaining = len(g.remaining_moves)
            if n_remaining <= 0:
                continue
            spa = max(1, sims_per_phase // n_remaining)
            g.sims_done = {m: 0 for m in g.remaining_moves}
            g._target_sims = spa  # type: ignore[attr-defined]

        # Run simulation rounds with virtual leaf batching
        any_work = True
        while any_work:
            any_work = False

            # ── Collect leaves across all games × actions × virtual leaves ──
            # Each leaf: (game_state, move, leaf_idx, leaf_board, path_or_None)
            leaves_to_eval: list[tuple[_GumbelGameState, chess.Move, int, object, object]] = []
            terminal_leaves: list[tuple[_GumbelGameState, int, object, object]] = []

            for g in games:
                target = getattr(g, '_target_sims', 1)
                for move in g.remaining_moves:
                    sims_left = target - g.sims_done[move]
                    if sims_left <= 0:
                        continue
                    any_work = True

                    child_idx = g.tree.get_child_for_move(move)
                    if child_idx is None:
                        g.sims_done[move] = target
                        continue

                    child_board = g.child_boards[move]
                    nvl = min(num_virtual_leaves, sims_left)

                    for _ in range(nvl):
                        if g.tree.is_terminal[child_idx]:
                            g.tree.backup(child_idx, terminal_value(child_board))
                            g.sims_done[move] += 1
                            continue

                        if not g.tree.is_expanded[child_idx]:
                            # Need to expand this child — add to eval batch
                            leaves_to_eval.append((g, move, child_idx, child_board, None))
                            g.sims_done[move] += 1
                            break  # Can't find more leaves until expanded

                        # Find leaf in subtree with virtual loss
                        leaf_idx, leaf_board, path = g.tree.find_leaf_in_subtree_vl(
                            child_idx, child_board, cpuct,
                        )

                        if g.tree.is_terminal[leaf_idx] or leaf_board.is_game_over(claim_draw=False):
                            g.tree.remove_virtual_loss(path)
                            g.tree.backup(leaf_idx, terminal_value(leaf_board))
                            g.sims_done[move] += 1
                        else:
                            leaves_to_eval.append((g, move, leaf_idx, leaf_board, path))
                            g.sims_done[move] += 1

            # ── Batch evaluate all collected leaves ──
            if leaves_to_eval:
                boards_to_eval = [lb for _, _, _, lb, _ in leaves_to_eval]
                policies_batch, values_batch = eval_batch_fn(boards_to_eval)

                for j, (g, move, leaf_idx, lb, path) in enumerate(leaves_to_eval):
                    policy = policies_batch[j]
                    value = float(values_batch[j])
                    g.tree.expand(leaf_idx, lb, policy)
                    if path is not None:
                        g.tree.remove_virtual_loss(path)
                    g.tree.backup(leaf_idx, value)

            # Check if all games have completed their sim budget this round
            if not leaves_to_eval:
                break

        # ── Halve remaining actions for each game ──
        for g in games:
            if len(g.remaining_moves) <= 1:
                continue
            q_values = g.tree.get_child_q_values(negate=True)
            g.remaining_moves.sort(
                key=lambda m: q_values.get(m, float('-inf')),
                reverse=True,
            )
            g.remaining_moves = g.remaining_moves[:max(1, len(g.remaining_moves) // 2)]

    return {g.game_idx: g.remaining_moves[0] for g in games}


def select_gumbel_move(
    candidate_moves: list[chess.Move],
    gumbel_scores: np.ndarray,
    q_values: dict[chess.Move, float],
    sigma: float,
) -> chess.Move:
    """Select move using Gumbel scores + Q-values.

    score(a) = gumbel_score(a) + sigma * Q(a)
    """
    best_score = float('-inf')
    best_move = candidate_moves[0]
    for i, move in enumerate(candidate_moves):
        q = q_values.get(move, 0.0)
        score = float(gumbel_scores[i]) + sigma * q if i < len(gumbel_scores) else sigma * q
        if score > best_score:
            best_score = score
            best_move = move
    return best_move


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
        """Evaluate one position with the neural network.

        Returns policy in actual-move space (unmirrored for black-to-move).
        """
        encoded = BoardEncoder.encode_board(board).unsqueeze(0).to(self.device)
        self.model.eval()
        policy_logits, values = _extract_policy_and_value(self.model(encoded))

        policy = torch.softmax(policy_logits.squeeze(0).float(), dim=0).cpu().numpy()
        value = values.squeeze(0).item()
        # Unmirror policy from model-space to actual-move space
        if board.turn == chess.BLACK:
            policy = mirror_policy(policy)
        return policy, value

    def search(self, board: chess.Board) -> dict[chess.Move, int]:
        """Run MCTS. Returns move visit counts."""
        if board.is_game_over(claim_draw=True):
            return {}

        tree = MCTSTree(capacity=50_000)
        policy, _ = self._evaluate(board)
        tree.expand(tree.root, board, policy)
        tree.add_dirichlet_noise(tree.root, self.config.dirichlet_alpha, self.config.dirichlet_epsilon)

        for _ in range(self.config.num_simulations):
            node, search_board = tree.find_leaf(board, self.config.cpuct)

            if tree.is_terminal[node] or search_board.is_game_over(claim_draw=False):
                value = terminal_value(search_board)
            else:
                policy, value = self._evaluate(search_board)
                tree.expand(node, search_board, policy)

            tree.backup(node, value)

        return tree.get_visit_counts()


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
        """Evaluate multiple positions in one forward pass.

        Returns policies in actual-move space (unmirrored for black-to-move).
        """
        if not boards:
            return np.empty((0, POLICY_SIZE)), np.empty(0)

        encoded = BoardEncoder.encode_board_batch(boards).to(self.device)
        self.model.eval()
        policy_logits, values = _extract_policy_and_value(self.model(encoded))

        policies = torch.softmax(policy_logits.float(), dim=-1).cpu().numpy()
        # Unmirror policies for black-to-move positions
        for i, b in enumerate(boards):
            if b.turn == chess.BLACK:
                policies[i] = mirror_policy(policies[i])
        return policies, values.cpu().numpy()

    def search_batch(
        self, boards: list[chess.Board],
    ) -> list[dict[chess.Move, int]]:
        """Run MCTS for multiple positions with batched evaluation + virtual loss."""
        n = len(boards)
        trees: list[MCTSTree] = [MCTSTree(capacity=50_000) for _ in range(n)]
        active = [i for i, b in enumerate(boards) if not b.is_game_over(claim_draw=True)]

        # Initial expansion
        if active:
            active_boards = [boards[i] for i in active]
            policies, _ = self._batched_evaluate(active_boards)
            for j, idx in enumerate(active):
                trees[idx].expand(trees[idx].root, boards[idx], policies[j])
                trees[idx].add_dirichlet_noise(
                    trees[idx].root,
                    self.config.dirichlet_alpha, self.config.dirichlet_epsilon,
                )

        # Simulations with virtual loss batching
        nvl = self.config.num_virtual_leaves

        for sim_start in range(0, self.config.num_simulations, nvl):
            leaves_per_game = min(nvl, self.config.num_simulations - sim_start)
            leaves: list[tuple[int, int, chess.Board, np.ndarray]] = []

            for i in active:
                for _ in range(leaves_per_game):
                    leaf_idx, search_board, path = trees[i].find_leaf_with_virtual_loss(
                        boards[i], self.config.cpuct,
                    )
                    if trees[i].is_terminal[leaf_idx] or search_board.is_game_over(claim_draw=False):
                        trees[i].remove_virtual_loss(path)
                        trees[i].backup(leaf_idx, terminal_value(search_board))
                    else:
                        leaves.append((i, leaf_idx, search_board, path))

            if leaves:
                leaf_boards = [lb for _, _, lb, _ in leaves]
                policies, values = self._batched_evaluate(leaf_boards)

                for j, (game_idx, leaf_idx, lb, path) in enumerate(leaves):
                    trees[game_idx].expand(leaf_idx, lb, policies[j])
                    trees[game_idx].remove_virtual_loss(path)
                    trees[game_idx].backup(leaf_idx, values[j])

        return [tree.get_visit_counts() for tree in trees]
