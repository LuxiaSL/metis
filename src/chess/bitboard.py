"""Numba-backed bitboard board for the self-play hot path.

The board keeps the minimum state needed by MCTS and encoding:
- 12 piece bitboards (white pieces first, then black)
- 64-entry mailbox for fast square lookups
- castling rights, en passant square, side to move, halfmove clock
- fixed-size move/state stack for push()/pop()

Hot operations such as move generation, attack tests, and push/pop are kept in
Numba-compiled helpers over NumPy arrays. The Python wrapper presents a small
subset of the ``python-chess`` API so the rest of the codebase can switch the
worker loop without dragging Python objects back into the hot path.
"""

from __future__ import annotations

from typing import Optional

import chess
import numba
import numpy as np

# ── Internal constants ──────────────────────────────────────────────────────

WHITE = 0
BLACK = 1
NO_SQUARE = -1

MAX_LEGAL_MOVES = 256
MAX_STACK = 1024

EMPTY = 0
WP = 1
WN = 2
WB = 3
WR = 4
WQ = 5
WK = 6
BP = 7
BN = 8
BB = 9
BR = 10
BQ = 11
BK = 12

WK_CASTLE = 1
WQ_CASTLE = 2
BK_CASTLE = 4
BQ_CASTLE = 8

PAWN = chess.PAWN
KNIGHT = chess.KNIGHT
BISHOP = chess.BISHOP
ROOK = chess.ROOK
QUEEN = chess.QUEEN
KING = chess.KING

_PIECE_CODE_TO_TYPE = np.array(
    [0, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
     PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING],
    dtype=np.int8,
)
_PIECE_CODE_TO_COLOR = np.array(
    [-1, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE,
     BLACK, BLACK, BLACK, BLACK, BLACK, BLACK],
    dtype=np.int8,
)

_KNIGHT_DELTAS = np.array(
    [
        (2, 1), (2, -1), (1, 2), (1, -2),
        (-1, 2), (-1, -2), (-2, 1), (-2, -1),
    ],
    dtype=np.int8,
)
_KING_DELTAS = np.array(
    [
        (1, 0), (1, 1), (0, 1), (-1, 1),
        (-1, 0), (-1, -1), (0, -1), (1, -1),
    ],
    dtype=np.int8,
)
_BISHOP_DIRS = np.array([(1, 1), (1, -1), (-1, 1), (-1, -1)], dtype=np.int8)
_ROOK_DIRS = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)], dtype=np.int8)
_QUEEN_PROMOS = np.array([QUEEN, ROOK, BISHOP, KNIGHT], dtype=np.int8)

_DARK_SQUARES = np.uint64(0)
for _sq in range(64):
    if ((chess.square_rank(_sq) + chess.square_file(_sq)) & 1) == 1:
        _DARK_SQUARES |= np.uint64(1) << np.uint64(_sq)
_LIGHT_SQUARES = np.uint64(((1 << 64) - 1) ^ int(_DARK_SQUARES))

_ZOBRIST_RNG = np.random.RandomState(20260407)
_ZOBRIST_PIECES = _ZOBRIST_RNG.randint(
    0, np.iinfo(np.uint64).max, size=(12, 64), dtype=np.uint64,
)
_ZOBRIST_CASTLING = _ZOBRIST_RNG.randint(
    0, np.iinfo(np.uint64).max, size=16, dtype=np.uint64,
)
_ZOBRIST_EP = _ZOBRIST_RNG.randint(
    0, np.iinfo(np.uint64).max, size=9, dtype=np.uint64,
)
_ZOBRIST_TURN = np.uint64(_ZOBRIST_RNG.randint(0, np.iinfo(np.uint64).max, dtype=np.uint64))


def _color_to_internal(color: bool) -> int:
    return WHITE if color == chess.WHITE else BLACK


@numba.njit(cache=True)
def _piece_code(piece_type: int, color: int) -> int:
    return piece_type if color == WHITE else piece_type + 6


def _piece_code_to_piece(code: int) -> Optional[chess.Piece]:
    if code == EMPTY:
        return None
    return chess.Piece(int(_PIECE_CODE_TO_TYPE[code]), color=code <= WK)


def _build_state_from_chess(board: chess.Board) -> tuple[np.ndarray, np.ndarray, np.uint8, np.int16, np.uint8, np.int16]:
    bitboards = np.zeros(12, dtype=np.uint64)
    mailbox = np.zeros(64, dtype=np.int8)

    for sq, piece in board.piece_map().items():
        color = WHITE if piece.color == chess.WHITE else BLACK
        code = _piece_code(piece.piece_type, color)
        bitboards[code - 1] |= np.uint64(1) << np.uint64(sq)
        mailbox[sq] = code

    castling = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castling |= WK_CASTLE
    if board.has_queenside_castling_rights(chess.WHITE):
        castling |= WQ_CASTLE
    if board.has_kingside_castling_rights(chess.BLACK):
        castling |= BK_CASTLE
    if board.has_queenside_castling_rights(chess.BLACK):
        castling |= BQ_CASTLE

    ep_square = np.int16(board.ep_square if board.ep_square is not None else NO_SQUARE)
    turn = np.uint8(WHITE if board.turn == chess.WHITE else BLACK)
    halfmove_clock = np.int16(board.halfmove_clock)
    return bitboards, mailbox, np.uint8(castling), ep_square, turn, halfmove_clock


_START_BITBOARDS, _START_MAILBOX, _START_CASTLING, _START_EP, _START_TURN, _START_HALFMOVE = _build_state_from_chess(chess.Board())


# ── Move encoding ───────────────────────────────────────────────────────────


@numba.njit(cache=True)
def encode_move(from_sq: int, to_sq: int, promotion: int = 0) -> int:
    return from_sq | (to_sq << 6) | (promotion << 12)


@numba.njit(cache=True)
def move_from_square(move: int) -> int:
    return move & 63


@numba.njit(cache=True)
def move_to_square(move: int) -> int:
    return (move >> 6) & 63


@numba.njit(cache=True)
def move_promotion(move: int) -> int:
    return (move >> 12) & 7


def move_from_chess_move(move: chess.Move) -> int:
    return int(encode_move(
        int(move.from_square),
        int(move.to_square),
        int(move.promotion or 0),
    ))


def move_to_chess_move(move: int) -> chess.Move:
    promo = int(move_promotion(move))
    return chess.Move(
        int(move_from_square(move)),
        int(move_to_square(move)),
        promotion=promo if promo else None,
    )


# ── Numba helpers ───────────────────────────────────────────────────────────


@numba.njit(cache=True)
def _square_rank(square: int) -> int:
    return square >> 3


@numba.njit(cache=True)
def _square_file(square: int) -> int:
    return square & 7


@numba.njit(cache=True)
def _is_own_piece(piece: int, color: int) -> bool:
    if piece == EMPTY:
        return False
    return _PIECE_CODE_TO_COLOR[piece] == color


@numba.njit(cache=True)
def _is_enemy_piece(piece: int, color: int) -> bool:
    if piece == EMPTY:
        return False
    return _PIECE_CODE_TO_COLOR[piece] != color


@numba.njit(cache=True)
def _add_move(out_moves: np.ndarray, count: int, from_sq: int, to_sq: int, promotion: int = 0) -> int:
    if count < len(out_moves):
        out_moves[count] = encode_move(from_sq, to_sq, promotion)
        return count + 1
    return count


@numba.njit(cache=True)
def _find_king_square(mailbox: np.ndarray, color: int) -> int:
    target = WK if color == WHITE else BK
    for sq in range(64):
        if mailbox[sq] == target:
            return sq
    return -1


@numba.njit(cache=True)
def _occupied_by(bitboards: np.ndarray, color: int) -> np.uint64:
    start = 0 if color == WHITE else 6
    occ = np.uint64(0)
    for i in range(start, start + 6):
        occ |= bitboards[i]
    return occ


@numba.njit(cache=True)
def _occupied_all(bitboards: np.ndarray) -> np.uint64:
    occ = np.uint64(0)
    for i in range(12):
        occ |= bitboards[i]
    return occ


@numba.njit(cache=True)
def _has_legal_en_passant(
    bitboards: np.ndarray,
    mailbox: np.ndarray,
    turn: int,
    castling_rights: int,
    ep_square: int,
    halfmove_clock: int,
) -> bool:
    if ep_square == NO_SQUARE:
        return False

    file = _square_file(ep_square)
    candidate_offsets = (-9, -7) if turn == WHITE else (7, 9)
    pawn_piece = WP if turn == WHITE else BP
    enemy = BLACK if turn == WHITE else WHITE
    tmp_bitboards = np.empty(12, dtype=np.uint64)
    tmp_mailbox = np.empty(64, dtype=np.int8)

    for delta in candidate_offsets:
        from_sq = ep_square + delta
        if not (0 <= from_sq < 64):
            continue
        if abs(_square_file(from_sq) - file) != 1:
            continue
        if mailbox[from_sq] != pawn_piece:
            continue

        tmp_bitboards[:] = bitboards
        tmp_mailbox[:] = mailbox
        _apply_move_inplace(
            tmp_bitboards, tmp_mailbox,
            encode_move(from_sq, ep_square, 0),
            turn, castling_rights, ep_square, halfmove_clock,
        )
        king_sq = _find_king_square(tmp_mailbox, turn)
        if king_sq >= 0 and not _is_square_attacked(tmp_bitboards, tmp_mailbox, king_sq, enemy):
            return True

    return False


@numba.njit(cache=True)
def _is_square_attacked(bitboards: np.ndarray, mailbox: np.ndarray, square: int, attacker_color: int) -> bool:
    rank = _square_rank(square)
    file = _square_file(square)

    if attacker_color == WHITE:
        if file > 0 and square >= 9 and mailbox[square - 9] == WP:
            return True
        if file < 7 and square >= 7 and mailbox[square - 7] == WP:
            return True
    else:
        if file > 0 and square <= 55 and mailbox[square + 7] == BP:
            return True
        if file < 7 and square <= 54 and mailbox[square + 9] == BP:
            return True

    knight_piece = WN if attacker_color == WHITE else BN
    for i in range(8):
        dr = int(_KNIGHT_DELTAS[i, 0])
        df = int(_KNIGHT_DELTAS[i, 1])
        nr = rank + dr
        nf = file + df
        if 0 <= nr < 8 and 0 <= nf < 8:
            if mailbox[nr * 8 + nf] == knight_piece:
                return True

    bishop_piece = WB if attacker_color == WHITE else BB
    queen_piece = WQ if attacker_color == WHITE else BQ
    for i in range(4):
        dr = int(_BISHOP_DIRS[i, 0])
        df = int(_BISHOP_DIRS[i, 1])
        nr = rank + dr
        nf = file + df
        while 0 <= nr < 8 and 0 <= nf < 8:
            piece = mailbox[nr * 8 + nf]
            if piece != EMPTY:
                if piece == bishop_piece or piece == queen_piece:
                    return True
                break
            nr += dr
            nf += df

    rook_piece = WR if attacker_color == WHITE else BR
    for i in range(4):
        dr = int(_ROOK_DIRS[i, 0])
        df = int(_ROOK_DIRS[i, 1])
        nr = rank + dr
        nf = file + df
        while 0 <= nr < 8 and 0 <= nf < 8:
            piece = mailbox[nr * 8 + nf]
            if piece != EMPTY:
                if piece == rook_piece or piece == queen_piece:
                    return True
                break
            nr += dr
            nf += df

    king_piece = WK if attacker_color == WHITE else BK
    for i in range(8):
        dr = int(_KING_DELTAS[i, 0])
        df = int(_KING_DELTAS[i, 1])
        nr = rank + dr
        nf = file + df
        if 0 <= nr < 8 and 0 <= nf < 8:
            if mailbox[nr * 8 + nf] == king_piece:
                return True

    return False


@numba.njit(cache=True)
def _apply_move_inplace(
    bitboards: np.ndarray,
    mailbox: np.ndarray,
    move: int,
    turn: int,
    castling_rights: int,
    ep_square: int,
    halfmove_clock: int,
) -> tuple[int, int, int, int]:
    from_sq = move_from_square(move)
    to_sq = move_to_square(move)
    promotion = move_promotion(move)

    moving_piece = int(mailbox[from_sq])
    moving_type = int(_PIECE_CODE_TO_TYPE[moving_piece])
    moving_color = int(_PIECE_CODE_TO_COLOR[moving_piece])
    captured_piece = int(mailbox[to_sq])
    capture = captured_piece != EMPTY

    from_mask = np.uint64(1) << np.uint64(from_sq)
    to_mask = np.uint64(1) << np.uint64(to_sq)

    mailbox[from_sq] = EMPTY
    bitboards[moving_piece - 1] &= ~from_mask

    is_ep = False
    if (moving_type == PAWN and to_sq == ep_square and captured_piece == EMPTY
            and _square_file(from_sq) != _square_file(to_sq)):
        is_ep = True
        cap_sq = to_sq - 8 if moving_color == WHITE else to_sq + 8
        cap_piece = int(mailbox[cap_sq])
        cap_mask = np.uint64(1) << np.uint64(cap_sq)
        mailbox[cap_sq] = EMPTY
        bitboards[cap_piece - 1] &= ~cap_mask
        captured_piece = cap_piece
        capture = True

    if captured_piece != EMPTY and not is_ep:
        bitboards[captured_piece - 1] &= ~to_mask

    placed_piece = moving_piece
    if promotion:
        placed_piece = _piece_code(promotion, moving_color)

    mailbox[to_sq] = placed_piece
    bitboards[placed_piece - 1] |= to_mask

    if moving_type == KING:
        if moving_color == WHITE:
            castling_rights &= ~(WK_CASTLE | WQ_CASTLE)
        else:
            castling_rights &= ~(BK_CASTLE | BQ_CASTLE)

        if abs(_square_file(to_sq) - _square_file(from_sq)) == 2:
            if moving_color == WHITE:
                if to_sq == chess.G1:
                    rook_from = chess.H1
                    rook_to = chess.F1
                else:
                    rook_from = chess.A1
                    rook_to = chess.D1
                rook_piece = WR
            else:
                if to_sq == chess.G8:
                    rook_from = chess.H8
                    rook_to = chess.F8
                else:
                    rook_from = chess.A8
                    rook_to = chess.D8
                rook_piece = BR

            rook_from_mask = np.uint64(1) << np.uint64(rook_from)
            rook_to_mask = np.uint64(1) << np.uint64(rook_to)
            mailbox[rook_from] = EMPTY
            mailbox[rook_to] = rook_piece
            bitboards[rook_piece - 1] &= ~rook_from_mask
            bitboards[rook_piece - 1] |= rook_to_mask

    if moving_type == ROOK:
        if from_sq == chess.H1:
            castling_rights &= ~WK_CASTLE
        elif from_sq == chess.A1:
            castling_rights &= ~WQ_CASTLE
        elif from_sq == chess.H8:
            castling_rights &= ~BK_CASTLE
        elif from_sq == chess.A8:
            castling_rights &= ~BQ_CASTLE

    if captured_piece == WR:
        if to_sq == chess.H1:
            castling_rights &= ~WK_CASTLE
        elif to_sq == chess.A1:
            castling_rights &= ~WQ_CASTLE
    elif captured_piece == BR:
        if to_sq == chess.H8:
            castling_rights &= ~BK_CASTLE
        elif to_sq == chess.A8:
            castling_rights &= ~BQ_CASTLE

    if moving_type == PAWN and abs(_square_rank(to_sq) - _square_rank(from_sq)) == 2:
        ep_square = (from_sq + to_sq) // 2
    else:
        ep_square = NO_SQUARE

    if moving_type == PAWN or capture:
        halfmove_clock = 0
    else:
        halfmove_clock += 1

    turn = BLACK if turn == WHITE else WHITE
    return int(castling_rights), int(ep_square), int(turn), int(halfmove_clock)


@numba.njit(cache=True)
def _push_with_snapshot(
    bitboards: np.ndarray,
    mailbox: np.ndarray,
    stack_bitboards: np.ndarray,
    stack_mailbox: np.ndarray,
    stack_move: np.ndarray,
    stack_castling: np.ndarray,
    stack_ep: np.ndarray,
    stack_turn: np.ndarray,
    stack_halfmove: np.ndarray,
    stack_len: int,
    castling_rights: int,
    ep_square: int,
    turn: int,
    halfmove_clock: int,
    move: int,
) -> tuple[int, int, int, int, int]:
    stack_bitboards[stack_len, :] = bitboards
    stack_mailbox[stack_len, :] = mailbox
    stack_move[stack_len] = move
    stack_castling[stack_len] = castling_rights
    stack_ep[stack_len] = ep_square
    stack_turn[stack_len] = turn
    stack_halfmove[stack_len] = halfmove_clock

    castling_rights, ep_square, turn, halfmove_clock = _apply_move_inplace(
        bitboards, mailbox, move, turn, castling_rights, ep_square, halfmove_clock,
    )
    return stack_len + 1, castling_rights, ep_square, turn, halfmove_clock


@numba.njit(cache=True)
def _pop_from_snapshot(
    bitboards: np.ndarray,
    mailbox: np.ndarray,
    stack_bitboards: np.ndarray,
    stack_mailbox: np.ndarray,
    stack_move: np.ndarray,
    stack_castling: np.ndarray,
    stack_ep: np.ndarray,
    stack_turn: np.ndarray,
    stack_halfmove: np.ndarray,
    stack_len: int,
) -> tuple[int, int, int, int, int, int]:
    stack_len -= 1
    bitboards[:] = stack_bitboards[stack_len]
    mailbox[:] = stack_mailbox[stack_len]
    return (
        stack_len,
        int(stack_move[stack_len]),
        int(stack_castling[stack_len]),
        int(stack_ep[stack_len]),
        int(stack_turn[stack_len]),
        int(stack_halfmove[stack_len]),
    )


@numba.njit(cache=True)
def _generate_pseudo_legal_moves(
    bitboards: np.ndarray,
    mailbox: np.ndarray,
    turn: int,
    castling_rights: int,
    ep_square: int,
    out_moves: np.ndarray,
) -> int:
    count = 0
    for from_sq in range(64):
        piece = int(mailbox[from_sq])
        if piece == EMPTY or int(_PIECE_CODE_TO_COLOR[piece]) != turn:
            continue

        piece_type = int(_PIECE_CODE_TO_TYPE[piece])
        rank = _square_rank(from_sq)
        file = _square_file(from_sq)

        if piece_type == PAWN:
            step = 8 if turn == WHITE else -8
            start_rank = 1 if turn == WHITE else 6
            promo_rank = 7 if turn == WHITE else 0
            target_rank = rank + (1 if turn == WHITE else -1)
            if 0 <= target_rank < 8:
                to_sq = from_sq + step
                if mailbox[to_sq] == EMPTY:
                    if target_rank == promo_rank:
                        for i in range(4):
                            count = _add_move(out_moves, count, from_sq, to_sq, int(_QUEEN_PROMOS[i]))
                    else:
                        count = _add_move(out_moves, count, from_sq, to_sq)
                        if rank == start_rank:
                            to_sq2 = from_sq + step * 2
                            if mailbox[to_sq2] == EMPTY:
                                count = _add_move(out_moves, count, from_sq, to_sq2)

            if turn == WHITE:
                capture_offsets = (7, 9)
            else:
                capture_offsets = (-9, -7)

            for delta in capture_offsets:
                to_sq = from_sq + delta
                if not (0 <= to_sq < 64):
                    continue
                tf = _square_file(to_sq)
                if abs(tf - file) != 1:
                    continue
                target = int(mailbox[to_sq])
                if _is_enemy_piece(target, turn) or to_sq == ep_square:
                    if _square_rank(to_sq) == promo_rank:
                        for i in range(4):
                            count = _add_move(out_moves, count, from_sq, to_sq, int(_QUEEN_PROMOS[i]))
                    else:
                        count = _add_move(out_moves, count, from_sq, to_sq)

        elif piece_type == KNIGHT:
            for i in range(8):
                dr = int(_KNIGHT_DELTAS[i, 0])
                df = int(_KNIGHT_DELTAS[i, 1])
                nr = rank + dr
                nf = file + df
                if 0 <= nr < 8 and 0 <= nf < 8:
                    to_sq = nr * 8 + nf
                    target = int(mailbox[to_sq])
                    if not _is_own_piece(target, turn):
                        count = _add_move(out_moves, count, from_sq, to_sq)

        elif piece_type == BISHOP or piece_type == QUEEN:
            for i in range(4):
                dr = int(_BISHOP_DIRS[i, 0])
                df = int(_BISHOP_DIRS[i, 1])
                nr = rank + dr
                nf = file + df
                while 0 <= nr < 8 and 0 <= nf < 8:
                    to_sq = nr * 8 + nf
                    target = int(mailbox[to_sq])
                    if _is_own_piece(target, turn):
                        break
                    count = _add_move(out_moves, count, from_sq, to_sq)
                    if target != EMPTY:
                        break
                    nr += dr
                    nf += df

        if piece_type == ROOK or piece_type == QUEEN:
            for i in range(4):
                dr = int(_ROOK_DIRS[i, 0])
                df = int(_ROOK_DIRS[i, 1])
                nr = rank + dr
                nf = file + df
                while 0 <= nr < 8 and 0 <= nf < 8:
                    to_sq = nr * 8 + nf
                    target = int(mailbox[to_sq])
                    if _is_own_piece(target, turn):
                        break
                    count = _add_move(out_moves, count, from_sq, to_sq)
                    if target != EMPTY:
                        break
                    nr += dr
                    nf += df

        elif piece_type == KING:
            for i in range(8):
                dr = int(_KING_DELTAS[i, 0])
                df = int(_KING_DELTAS[i, 1])
                nr = rank + dr
                nf = file + df
                if 0 <= nr < 8 and 0 <= nf < 8:
                    to_sq = nr * 8 + nf
                    target = int(mailbox[to_sq])
                    if not _is_own_piece(target, turn):
                        count = _add_move(out_moves, count, from_sq, to_sq)

            enemy = BLACK if turn == WHITE else WHITE
            if turn == WHITE and from_sq == chess.E1 and mailbox[chess.E1] == WK:
                if (castling_rights & WK_CASTLE and mailbox[chess.F1] == EMPTY and mailbox[chess.G1] == EMPTY
                        and mailbox[chess.H1] == WR
                        and not _is_square_attacked(bitboards, mailbox, chess.E1, enemy)
                        and not _is_square_attacked(bitboards, mailbox, chess.F1, enemy)
                        and not _is_square_attacked(bitboards, mailbox, chess.G1, enemy)):
                    count = _add_move(out_moves, count, chess.E1, chess.G1)
                if (castling_rights & WQ_CASTLE and mailbox[chess.B1] == EMPTY and mailbox[chess.C1] == EMPTY
                        and mailbox[chess.D1] == EMPTY and mailbox[chess.A1] == WR
                        and not _is_square_attacked(bitboards, mailbox, chess.E1, enemy)
                        and not _is_square_attacked(bitboards, mailbox, chess.D1, enemy)
                        and not _is_square_attacked(bitboards, mailbox, chess.C1, enemy)):
                    count = _add_move(out_moves, count, chess.E1, chess.C1)
            elif turn == BLACK and from_sq == chess.E8 and mailbox[chess.E8] == BK:
                if (castling_rights & BK_CASTLE and mailbox[chess.F8] == EMPTY and mailbox[chess.G8] == EMPTY
                        and mailbox[chess.H8] == BR
                        and not _is_square_attacked(bitboards, mailbox, chess.E8, enemy)
                        and not _is_square_attacked(bitboards, mailbox, chess.F8, enemy)
                        and not _is_square_attacked(bitboards, mailbox, chess.G8, enemy)):
                    count = _add_move(out_moves, count, chess.E8, chess.G8)
                if (castling_rights & BQ_CASTLE and mailbox[chess.B8] == EMPTY and mailbox[chess.C8] == EMPTY
                        and mailbox[chess.D8] == EMPTY and mailbox[chess.A8] == BR
                        and not _is_square_attacked(bitboards, mailbox, chess.E8, enemy)
                        and not _is_square_attacked(bitboards, mailbox, chess.D8, enemy)
                        and not _is_square_attacked(bitboards, mailbox, chess.C8, enemy)):
                    count = _add_move(out_moves, count, chess.E8, chess.C8)

    return count


@numba.njit(cache=True)
def _generate_legal_moves(
    bitboards: np.ndarray,
    mailbox: np.ndarray,
    turn: int,
    castling_rights: int,
    ep_square: int,
    halfmove_clock: int,
    out_moves: np.ndarray,
) -> int:
    pseudo = np.empty(MAX_LEGAL_MOVES, dtype=np.int32)
    tmp_bitboards = np.empty(12, dtype=np.uint64)
    tmp_mailbox = np.empty(64, dtype=np.int8)

    pseudo_count = _generate_pseudo_legal_moves(
        bitboards, mailbox, turn, castling_rights, ep_square, pseudo,
    )

    legal_count = 0
    enemy = BLACK if turn == WHITE else WHITE
    for i in range(pseudo_count):
        tmp_bitboards[:] = bitboards
        tmp_mailbox[:] = mailbox
        _apply_move_inplace(
            tmp_bitboards, tmp_mailbox, int(pseudo[i]),
            turn, castling_rights, ep_square, halfmove_clock,
        )
        king_sq = _find_king_square(tmp_mailbox, turn)
        if king_sq >= 0 and not _is_square_attacked(tmp_bitboards, tmp_mailbox, king_sq, enemy):
            out_moves[legal_count] = pseudo[i]
            legal_count += 1

    return legal_count


@numba.njit(cache=True)
def _compute_hash(
    bitboards: np.ndarray,
    mailbox: np.ndarray,
    turn: int,
    castling_rights: int,
    ep_square: int,
    halfmove_clock: int,
) -> np.uint64:
    h = np.uint64(0)
    for piece_idx in range(12):
        bb = bitboards[piece_idx]
        for sq in range(64):
            if bb & (np.uint64(1) << np.uint64(sq)):
                h ^= _ZOBRIST_PIECES[piece_idx, sq]
        # No lsb scanning here; 12x64 checks is acceptable relative to push/pop frequency.
    h ^= _ZOBRIST_CASTLING[castling_rights]
    if ep_square == NO_SQUARE or not _has_legal_en_passant(
        bitboards, mailbox, turn, castling_rights, ep_square, halfmove_clock,
    ):
        h ^= _ZOBRIST_EP[8]
    else:
        h ^= _ZOBRIST_EP[_square_file(ep_square)]
    if turn == WHITE:
        h ^= _ZOBRIST_TURN
    return h


@numba.njit(cache=True)
def _repetition_count(hash_history: np.ndarray, hash_len: int, halfmove_clock: int) -> int:
    if hash_len <= 0:
        return 0
    current = hash_history[hash_len - 1]
    count = 0
    start = hash_len - 1 - halfmove_clock
    if start < 0:
        start = 0
    for i in range(hash_len - 1, start - 1, -2):
        if hash_history[i] == current:
            count += 1
    return count


@numba.njit(cache=True)
def _is_insufficient_material(bitboards: np.ndarray) -> bool:
    if bitboards[WP - 1] | bitboards[BP - 1] | bitboards[WR - 1] | bitboards[BR - 1] | bitboards[WQ - 1] | bitboards[BQ - 1]:
        return False

    knight_count = 0
    for sq in range(64):
        mask = np.uint64(1) << np.uint64(sq)
        if bitboards[WN - 1] & mask:
            knight_count += 1
        if bitboards[BN - 1] & mask:
            knight_count += 1

    bishops = bitboards[WB - 1] | bitboards[BB - 1]
    if knight_count > 0:
        total_pieces = 2 + knight_count
        for sq in range(64):
            mask = np.uint64(1) << np.uint64(sq)
            if bishops & mask:
                total_pieces += 1
        return total_pieces == 3

    if bishops:
        return (bishops & _DARK_SQUARES) == 0 or (bishops & _LIGHT_SQUARES) == 0

    return True


@numba.njit(cache=True)
def _compute_outcome_code(
    bitboards: np.ndarray,
    mailbox: np.ndarray,
    turn: int,
    castling_rights: int,
    ep_square: int,
    halfmove_clock: int,
    hash_history: np.ndarray,
    hash_len: int,
    claim_draw: bool,
) -> int:
    legal_buf = np.empty(MAX_LEGAL_MOVES, dtype=np.int32)
    legal_count = _generate_legal_moves(
        bitboards, mailbox, turn, castling_rights, ep_square, halfmove_clock, legal_buf,
    )

    king_sq = _find_king_square(mailbox, turn)
    enemy = BLACK if turn == WHITE else WHITE
    in_check = king_sq >= 0 and _is_square_attacked(bitboards, mailbox, king_sq, enemy)

    if legal_count == 0 and in_check:
        return 2 if turn == WHITE else 1
    if _is_insufficient_material(bitboards):
        return 3
    if legal_count == 0:
        return 3
    if halfmove_clock >= 150:
        return 3

    repetition_count = _repetition_count(hash_history, hash_len, halfmove_clock)
    if repetition_count >= 5:
        return 3
    if claim_draw and halfmove_clock >= 100:
        return 3
    if claim_draw and repetition_count >= 3:
        return 3
    return 0


# ── Python wrapper ─────────────────────────────────────────────────────────


class SquareSet:
    """Small iterable wrapper compatible with the parts of python-chess we use."""

    __slots__ = ["_bitboard"]

    def __init__(self, bitboard: np.uint64) -> None:
        self._bitboard = np.uint64(bitboard)

    def __iter__(self):
        bb = int(self._bitboard)
        while bb:
            lsb = bb & -bb
            yield lsb.bit_length() - 1
            bb ^= lsb

    def __len__(self) -> int:
        return int(int(self._bitboard).bit_count())

    def __bool__(self) -> bool:
        return int(self._bitboard) != 0

    def __int__(self) -> int:
        return int(self._bitboard)


class LegalMoveList:
    """Lazy iterable over legal moves backed by encoded move integers."""

    __slots__ = ["_moves"]

    def __init__(self, moves: np.ndarray) -> None:
        self._moves = moves

    def __iter__(self):
        for move in self._moves:
            yield move_to_chess_move(int(move))

    def __len__(self) -> int:
        return int(len(self._moves))

    def __bool__(self) -> bool:
        return len(self._moves) > 0


class Board:
    """Bitboard-backed board with a small python-chess-compatible facade."""

    __slots__ = [
        "_bitboards", "_mailbox",
        "_castling_rights", "_ep_square", "_turn", "_halfmove_clock",
        "_stack_bitboards", "_stack_mailbox", "_stack_move",
        "_stack_castling", "_stack_ep", "_stack_turn", "_stack_halfmove",
        "_stack_len",
        "_hash_history", "_hash_len",
        "_legal_buf", "_legal_moves_cache",
    ]

    def __init__(self) -> None:
        self._bitboards = _START_BITBOARDS.copy()
        self._mailbox = _START_MAILBOX.copy()
        self._castling_rights = int(_START_CASTLING)
        self._ep_square = int(_START_EP)
        self._turn = int(_START_TURN)
        self._halfmove_clock = int(_START_HALFMOVE)

        self._stack_bitboards = np.zeros((MAX_STACK, 12), dtype=np.uint64)
        self._stack_mailbox = np.zeros((MAX_STACK, 64), dtype=np.int8)
        self._stack_move = np.zeros(MAX_STACK, dtype=np.int32)
        self._stack_castling = np.zeros(MAX_STACK, dtype=np.uint8)
        self._stack_ep = np.full(MAX_STACK, NO_SQUARE, dtype=np.int16)
        self._stack_turn = np.zeros(MAX_STACK, dtype=np.uint8)
        self._stack_halfmove = np.zeros(MAX_STACK, dtype=np.int16)
        self._stack_len = 0

        self._hash_history = np.zeros(MAX_STACK + 1, dtype=np.uint64)
        self._hash_len = 1
        self._hash_history[0] = _compute_hash(
            self._bitboards, self._mailbox, self._turn,
            self._castling_rights, self._ep_square, self._halfmove_clock,
        )

        self._legal_buf = np.empty(MAX_LEGAL_MOVES, dtype=np.int32)
        self._legal_moves_cache: Optional[np.ndarray] = None

    @classmethod
    def from_chess_board(cls, board: chess.Board) -> Board:
        instance = cls.__new__(cls)
        instance._bitboards, instance._mailbox, castling, ep, turn, halfmove = _build_state_from_chess(board)
        instance._castling_rights = int(castling)
        instance._ep_square = int(ep)
        instance._turn = int(turn)
        instance._halfmove_clock = int(halfmove)
        instance._stack_bitboards = np.zeros((MAX_STACK, 12), dtype=np.uint64)
        instance._stack_mailbox = np.zeros((MAX_STACK, 64), dtype=np.int8)
        instance._stack_move = np.zeros(MAX_STACK, dtype=np.int32)
        instance._stack_castling = np.zeros(MAX_STACK, dtype=np.uint8)
        instance._stack_ep = np.full(MAX_STACK, NO_SQUARE, dtype=np.int16)
        instance._stack_turn = np.zeros(MAX_STACK, dtype=np.uint8)
        instance._stack_halfmove = np.zeros(MAX_STACK, dtype=np.int16)
        instance._stack_len = 0
        instance._hash_history = np.zeros(MAX_STACK + 1, dtype=np.uint64)
        instance._hash_len = 1
        instance._hash_history[0] = _compute_hash(
            instance._bitboards, instance._mailbox, instance._turn,
            instance._castling_rights, instance._ep_square, instance._halfmove_clock,
        )
        instance._legal_buf = np.empty(MAX_LEGAL_MOVES, dtype=np.int32)
        instance._legal_moves_cache = None
        return instance

    @classmethod
    def from_fen(cls, fen: str) -> Board:
        return cls.from_chess_board(chess.Board(fen))

    def copy(self, *, stack: bool = True) -> Board:
        other = self.__class__.__new__(self.__class__)
        other._bitboards = self._bitboards.copy()
        other._mailbox = self._mailbox.copy()
        other._castling_rights = self._castling_rights
        other._ep_square = self._ep_square
        other._turn = self._turn
        other._halfmove_clock = self._halfmove_clock
        other._stack_bitboards = np.zeros((MAX_STACK, 12), dtype=np.uint64)
        other._stack_mailbox = np.zeros((MAX_STACK, 64), dtype=np.int8)
        other._stack_move = np.zeros(MAX_STACK, dtype=np.int32)
        other._stack_castling = np.zeros(MAX_STACK, dtype=np.uint8)
        other._stack_ep = np.full(MAX_STACK, NO_SQUARE, dtype=np.int16)
        other._stack_turn = np.zeros(MAX_STACK, dtype=np.uint8)
        other._stack_halfmove = np.zeros(MAX_STACK, dtype=np.int16)
        other._stack_len = 0
        other._hash_history = np.zeros(MAX_STACK + 1, dtype=np.uint64)
        other._legal_buf = np.empty(MAX_LEGAL_MOVES, dtype=np.int32)
        other._legal_moves_cache = None

        if stack:
            other._stack_bitboards[:self._stack_len] = self._stack_bitboards[:self._stack_len]
            other._stack_mailbox[:self._stack_len] = self._stack_mailbox[:self._stack_len]
            other._stack_move[:self._stack_len] = self._stack_move[:self._stack_len]
            other._stack_castling[:self._stack_len] = self._stack_castling[:self._stack_len]
            other._stack_ep[:self._stack_len] = self._stack_ep[:self._stack_len]
            other._stack_turn[:self._stack_len] = self._stack_turn[:self._stack_len]
            other._stack_halfmove[:self._stack_len] = self._stack_halfmove[:self._stack_len]
            other._stack_len = self._stack_len
            other._hash_history[:self._hash_len] = self._hash_history[:self._hash_len]
            other._hash_len = self._hash_len
        else:
            other._hash_len = 1
            other._hash_history[0] = _compute_hash(
                other._bitboards, other._mailbox, other._turn,
                other._castling_rights, other._ep_square, other._halfmove_clock,
            )

        return other

    @property
    def turn(self) -> bool:
        return chess.WHITE if self._turn == WHITE else chess.BLACK

    @property
    def castling_rights(self) -> int:
        return self._castling_rights

    @property
    def ep_square(self) -> Optional[int]:
        return None if self._ep_square == NO_SQUARE else self._ep_square

    @property
    def halfmove_clock(self) -> int:
        return self._halfmove_clock

    @property
    def legal_moves(self) -> LegalMoveList:
        return LegalMoveList(self.legal_move_codes())

    def legal_move_codes(self) -> np.ndarray:
        if self._legal_moves_cache is None:
            count = _generate_legal_moves(
                self._bitboards, self._mailbox, self._turn,
                self._castling_rights, self._ep_square, self._halfmove_clock,
                self._legal_buf,
            )
            self._legal_moves_cache = self._legal_buf[:count].copy()
        return self._legal_moves_cache

    def legal_move_count(self) -> int:
        return int(len(self.legal_move_codes()))

    def push(self, move: int | chess.Move) -> None:
        if self._stack_len >= MAX_STACK:
            raise RuntimeError(f"Bitboard move stack exhausted ({MAX_STACK})")

        move_code = int(move) if isinstance(move, (int, np.integer)) else move_from_chess_move(move)
        (
            self._stack_len,
            self._castling_rights,
            self._ep_square,
            self._turn,
            self._halfmove_clock,
        ) = _push_with_snapshot(
            self._bitboards,
            self._mailbox,
            self._stack_bitboards,
            self._stack_mailbox,
            self._stack_move,
            self._stack_castling,
            self._stack_ep,
            self._stack_turn,
            self._stack_halfmove,
            self._stack_len,
            self._castling_rights,
            self._ep_square,
            self._turn,
            self._halfmove_clock,
            move_code,
        )
        self._hash_history[self._hash_len] = _compute_hash(
            self._bitboards, self._mailbox, self._turn,
            self._castling_rights, self._ep_square, self._halfmove_clock,
        )
        self._hash_len += 1
        self._legal_moves_cache = None

    def pop(self) -> chess.Move:
        if self._stack_len <= 0:
            raise IndexError("pop from empty move stack")

        if self._legal_moves_cache is not None:
            self._legal_moves_cache = None

        (
            self._stack_len,
            move_code,
            self._castling_rights,
            self._ep_square,
            self._turn,
            self._halfmove_clock,
        ) = _pop_from_snapshot(
            self._bitboards,
            self._mailbox,
            self._stack_bitboards,
            self._stack_mailbox,
            self._stack_move,
            self._stack_castling,
            self._stack_ep,
            self._stack_turn,
            self._stack_halfmove,
            self._stack_len,
        )
        self._hash_len -= 1
        return move_to_chess_move(int(move_code))

    def push_uci(self, uci: str) -> None:
        self.push(chess.Move.from_uci(uci))

    def has_kingside_castling_rights(self, color: bool) -> bool:
        if color == chess.WHITE:
            return bool(self._castling_rights & WK_CASTLE)
        return bool(self._castling_rights & BK_CASTLE)

    def has_queenside_castling_rights(self, color: bool) -> bool:
        if color == chess.WHITE:
            return bool(self._castling_rights & WQ_CASTLE)
        return bool(self._castling_rights & BQ_CASTLE)

    def pieces(self, piece_type: int, color: bool) -> SquareSet:
        idx = _piece_code(piece_type, _color_to_internal(color)) - 1
        return SquareSet(self._bitboards[idx])

    def piece_at(self, square: int) -> Optional[chess.Piece]:
        return _piece_code_to_piece(int(self._mailbox[square]))

    def is_game_over(self, *, claim_draw: bool = False) -> bool:
        return self._outcome_code(claim_draw) != 0

    def result(self, *, claim_draw: bool = False) -> str:
        outcome = self._outcome_code(claim_draw)
        if outcome == 1:
            return "1-0"
        if outcome == 2:
            return "0-1"
        if outcome == 3:
            return "1/2-1/2"
        return "*"

    def _outcome_code(self, claim_draw: bool) -> int:
        return int(_compute_outcome_code(
            self._bitboards,
            self._mailbox,
            self._turn,
            self._castling_rights,
            self._ep_square,
            self._halfmove_clock,
            self._hash_history,
            self._hash_len,
            claim_draw,
        ))
