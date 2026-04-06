"""Board and move encoding for the chess transformer.

Board encoding: 67 tokens (3 global + 64 squares)
Move encoding: AlphaZero 8x8x73 = 4672 policy outputs

Global tokens:
  [0] castling rights (0-15, 4-bit packed)
  [1] en passant file (0-7, or 8 for none)
  [2] side to move (0=white, 1=black)

Square tokens (indices 3-66):
  Piece type index 0-12: empty, P, N, B, R, Q, K, p, n, b, r, q, k
  Ordered by square index (0=a1, 1=b1, ..., 63=h8)

Move types per square (73 total):
  0-55:  Queen-type moves (8 directions x 7 distances)
  56-63: Knight moves (8 L-shaped offsets)
  64-72: Underpromotions (3 pieces x 3 directions)
"""

from __future__ import annotations

from typing import Optional

import chess
import torch

# ── Piece encoding ─────────────────────────────────────────────────────────

PIECE_TO_INDEX: dict[Optional[chess.Piece], int] = {None: 0}
for _color in [chess.WHITE, chess.BLACK]:
    for _pt in range(1, 7):  # PAWN=1 .. KING=6
        _piece = chess.Piece(_pt, _color)
        PIECE_TO_INDEX[_piece] = _pt if _color == chess.WHITE else _pt + 6

NUM_PIECE_TYPES = 13
NUM_SQUARES = 64
NUM_GLOBAL_TOKENS = 3
SEQ_LEN = NUM_SQUARES + NUM_GLOBAL_TOKENS  # 67

# ── Move encoding constants ────────────────────────────────────────────────

NUM_MOVE_TYPES = 73
POLICY_SIZE = NUM_SQUARES * NUM_MOVE_TYPES  # 4672

# Queen-type directions: (delta_rank, delta_file)
QUEEN_DIRS: list[tuple[int, int]] = [
    (1, 0),    # 0: N
    (1, 1),    # 1: NE
    (0, 1),    # 2: E
    (-1, 1),   # 3: SE
    (-1, 0),   # 4: S
    (-1, -1),  # 5: SW
    (0, -1),   # 6: W
    (1, -1),   # 7: NW
]

# Knight offsets: (delta_rank, delta_file)
KNIGHT_OFFSETS: list[tuple[int, int]] = [
    (2, 1), (2, -1), (1, 2), (1, -2),
    (-1, 2), (-1, -2), (-2, 1), (-2, -1),
]

# Underpromotion piece ordering
UNDERPROMO_PIECES: list[int] = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
UNDERPROMO_PIECE_TO_IDX: dict[int, int] = {
    chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2,
}


# ── Move type computation ──────────────────────────────────────────────────

def _compute_move_type(
    from_sq: int,
    to_sq: int,
    promotion: Optional[int] = None,
) -> int:
    """Compute the move type index (0-72) from from/to squares and promotion.

    Underpromotions (knight/bishop/rook) use move types 64-72.
    Queen promotions are encoded as the regular queen-type move (distance 1).
    Knight moves are identified by their L-shape geometry.
    All other moves use queen-type encoding (direction x distance).
    """
    dr = chess.square_rank(to_sq) - chess.square_rank(from_sq)
    df = chess.square_file(to_sq) - chess.square_file(from_sq)

    # Underpromotion takes priority (before knight-shape check)
    if promotion is not None and promotion != chess.QUEEN:
        piece_idx = UNDERPROMO_PIECE_TO_IDX[promotion]
        dir_idx = df + 1  # -1 -> 0 (left), 0 -> 1 (forward), +1 -> 2 (right)
        return 64 + piece_idx * 3 + dir_idx

    # Knight move: L-shape (neither straight nor diagonal)
    if dr != 0 and df != 0 and abs(dr) != abs(df):
        for i, (kr, kf) in enumerate(KNIGHT_OFFSETS):
            if dr == kr and df == kf:
                return 56 + i
        raise ValueError(f"Invalid knight move delta: ({dr}, {df})")

    # Queen-type move (straight or diagonal)
    if dr > 0 and df == 0:
        direction = 0   # N
    elif dr > 0 and df > 0:
        direction = 1   # NE
    elif dr == 0 and df > 0:
        direction = 2   # E
    elif dr < 0 and df > 0:
        direction = 3   # SE
    elif dr < 0 and df == 0:
        direction = 4   # S
    elif dr < 0 and df < 0:
        direction = 5   # SW
    elif dr == 0 and df < 0:
        direction = 6   # W
    elif dr > 0 and df < 0:
        direction = 7   # NW
    else:
        raise ValueError(f"Invalid move delta: ({dr}, {df})")

    distance = max(abs(dr), abs(df))
    return direction * 7 + (distance - 1)


def _build_index_to_move() -> list[Optional[chess.Move]]:
    """Build reverse mapping from policy index to chess.Move."""
    index_to_move: list[Optional[chess.Move]] = [None] * POLICY_SIZE

    for from_sq in range(64):
        from_rank = chess.square_rank(from_sq)
        from_file = chess.square_file(from_sq)

        # Queen-type moves (distance 1-7 in 8 directions)
        for dir_idx, (dr, df) in enumerate(QUEEN_DIRS):
            for dist in range(1, 8):
                to_rank = from_rank + dr * dist
                to_file = from_file + df * dist
                if 0 <= to_rank < 8 and 0 <= to_file < 8:
                    to_sq = chess.square(to_file, to_rank)
                    move_type = dir_idx * 7 + (dist - 1)
                    idx = from_sq * NUM_MOVE_TYPES + move_type
                    index_to_move[idx] = chess.Move(from_sq, to_sq)

        # Knight moves
        for knight_idx, (dr, df) in enumerate(KNIGHT_OFFSETS):
            to_rank = from_rank + dr
            to_file = from_file + df
            if 0 <= to_rank < 8 and 0 <= to_file < 8:
                to_sq = chess.square(to_file, to_rank)
                move_type = 56 + knight_idx
                idx = from_sq * NUM_MOVE_TYPES + move_type
                index_to_move[idx] = chess.Move(from_sq, to_sq)

        # Underpromotions (knight/bishop/rook x left/forward/right)
        for piece_idx, promo_piece in enumerate(UNDERPROMO_PIECES):
            for dir_idx, delta_f in enumerate([-1, 0, 1]):
                for src_rank, dst_rank in [(6, 7), (1, 0)]:
                    if from_rank == src_rank:
                        to_file = from_file + delta_f
                        if 0 <= to_file < 8:
                            to_sq = chess.square(to_file, dst_rank)
                            move_type = 64 + piece_idx * 3 + dir_idx
                            idx = from_sq * NUM_MOVE_TYPES + move_type
                            index_to_move[idx] = chess.Move(
                                from_sq, to_sq, promotion=promo_piece,
                            )

    return index_to_move


# Precomputed at import time
INDEX_TO_MOVE: list[Optional[chess.Move]] = _build_index_to_move()


# ── Public API ─────────────────────────────────────────────────────────────


class BoardEncoder:
    """Encodes chess.Board into tensor representation for the transformer."""

    @staticmethod
    def encode_board(board: chess.Board) -> torch.Tensor:
        """Encode a board position as a tensor of token indices.

        Returns:
            Tensor of shape (67,) dtype=long.
        """
        tokens = torch.zeros(SEQ_LEN, dtype=torch.long)

        # Global token 0: castling rights (4-bit packed)
        castling = 0
        if board.has_kingside_castling_rights(chess.WHITE):
            castling |= 1
        if board.has_queenside_castling_rights(chess.WHITE):
            castling |= 2
        if board.has_kingside_castling_rights(chess.BLACK):
            castling |= 4
        if board.has_queenside_castling_rights(chess.BLACK):
            castling |= 8
        tokens[0] = castling

        # Global token 1: en passant file (0-7) or 8 (none)
        tokens[1] = chess.square_file(board.ep_square) if board.ep_square is not None else 8

        # Global token 2: side to move
        tokens[2] = 0 if board.turn == chess.WHITE else 1

        # Square tokens via piece_map (faster than iterating all 64)
        for sq, piece in board.piece_map().items():
            tokens[3 + sq] = PIECE_TO_INDEX[piece]

        return tokens

    @staticmethod
    def encode_board_batch(boards: list[chess.Board]) -> torch.Tensor:
        """Encode multiple boards. Returns Tensor shape (B, 67)."""
        return torch.stack([BoardEncoder.encode_board(b) for b in boards])


class MoveEncoder:
    """Encodes chess moves using AlphaZero's 8x8x73 policy format."""

    @staticmethod
    def move_to_index(move: chess.Move) -> int:
        """Convert a chess.Move to a policy index (0-4671)."""
        move_type = _compute_move_type(
            move.from_square, move.to_square, move.promotion,
        )
        return move.from_square * NUM_MOVE_TYPES + move_type

    @staticmethod
    def index_to_move(index: int) -> Optional[chess.Move]:
        """Convert a policy index to a chess.Move (or None if invalid)."""
        if 0 <= index < POLICY_SIZE:
            return INDEX_TO_MOVE[index]
        return None

    @staticmethod
    def legal_move_mask(board: chess.Board) -> torch.Tensor:
        """Create boolean mask over policy space for legal moves.

        Returns:
            Tensor shape (4672,) dtype=bool.
        """
        mask = torch.zeros(POLICY_SIZE, dtype=torch.bool)
        for move in board.legal_moves:
            try:
                mask[MoveEncoder.move_to_index(move)] = True
            except ValueError:
                pass  # shouldn't happen for standard chess
        return mask

    @staticmethod
    def encode_policy(visit_counts: dict[chess.Move, int]) -> torch.Tensor:
        """Convert MCTS visit counts to a normalized policy distribution.

        Returns:
            Tensor shape (4672,) dtype=float32, sums to 1.
        """
        policy = torch.zeros(POLICY_SIZE, dtype=torch.float32)
        total = sum(visit_counts.values())
        if total == 0:
            return policy
        for move, count in visit_counts.items():
            try:
                idx = MoveEncoder.move_to_index(move)
                policy[idx] = count / total
            except ValueError:
                pass
        return policy

    @staticmethod
    def decode_move(
        policy_logits: torch.Tensor,
        board: chess.Board,
        temperature: float = 1.0,
    ) -> chess.Move:
        """Select a move from policy logits with legal-move masking.

        Args:
            policy_logits: Raw logits shape (4672,).
            board: Current position (for legal move filtering).
            temperature: 0 = greedy, >0 = softmax sampling.

        Returns:
            Selected legal chess.Move.

        Raises:
            ValueError: If no legal moves available.
        """
        mask = MoveEncoder.legal_move_mask(board)
        if not mask.any():
            raise ValueError("No legal moves available")

        masked_logits = policy_logits.float().clone()
        masked_logits[~mask] = float("-inf")

        if temperature == 0:
            idx = masked_logits.argmax().item()
        else:
            probs = torch.softmax(masked_logits / temperature, dim=0)
            idx = torch.multinomial(probs, 1).item()

        move = INDEX_TO_MOVE[idx]
        if move is None:
            raise ValueError(f"Invalid move index: {idx}")

        # Queen-type moves landing on the last rank with a pawn need queen promotion
        if move.promotion is None:
            piece = board.piece_at(move.from_square)
            if piece is not None and piece.piece_type == chess.PAWN:
                to_rank = chess.square_rank(move.to_square)
                if to_rank in (0, 7):
                    move = chess.Move(
                        move.from_square, move.to_square, promotion=chess.QUEEN,
                    )

        return move
