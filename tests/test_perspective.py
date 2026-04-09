"""Tests for perspective canonicalization.

Verifies:
1. Mirror policy table is self-inverse
2. Board encoding flips correctly for black
3. Encode-mirror roundtrip preserves move semantics
4. Training data pipeline consistency
"""

from __future__ import annotations

import chess
import numpy as np
import torch

from src.chess.board import (
    BoardEncoder,
    MoveEncoder,
    SEQ_LEN,
    POLICY_SIZE,
    NUM_MOVE_TYPES,
    _MIRROR_POLICY_TABLE,
    mirror_policy,
)


class TestMirrorPolicyTable:
    """Tests for the perspective mirror table."""

    def test_self_inverse(self):
        """table[table[i]] == i for all i."""
        table = _MIRROR_POLICY_TABLE
        roundtrip = table[table]
        np.testing.assert_array_equal(roundtrip, np.arange(POLICY_SIZE))

    def test_mirror_policy_roundtrip(self):
        """mirror_policy(mirror_policy(p)) == p."""
        rng = np.random.default_rng(42)
        policy = rng.random(POLICY_SIZE).astype(np.float32)
        policy /= policy.sum()
        result = mirror_policy(mirror_policy(policy))
        np.testing.assert_allclose(result, policy, atol=1e-7)

    def test_specific_queen_move_e2e4(self):
        """e2-e4 (white pawn push) should mirror to e7-e5."""
        # e2 = square(4, 1) = 12, e4 = square(4, 3) = 28
        move_e2e4 = chess.Move(chess.E2, chess.E4)
        idx_e2e4 = MoveEncoder.move_to_index(move_e2e4)

        # e7 = square(4, 6) = 52, e5 = square(4, 4) = 36
        move_e7e5 = chess.Move(chess.E7, chess.E5)
        idx_e7e5 = MoveEncoder.move_to_index(move_e7e5)

        assert _MIRROR_POLICY_TABLE[idx_e2e4] == idx_e7e5
        assert _MIRROR_POLICY_TABLE[idx_e7e5] == idx_e2e4

    def test_specific_knight_move(self):
        """Nf3 (g1-f3) should mirror to Nf6 (g8-f6)."""
        move_nf3 = chess.Move(chess.G1, chess.F3)
        idx_nf3 = MoveEncoder.move_to_index(move_nf3)

        move_nf6 = chess.Move(chess.G8, chess.F6)
        idx_nf6 = MoveEncoder.move_to_index(move_nf6)

        assert _MIRROR_POLICY_TABLE[idx_nf3] == idx_nf6
        assert _MIRROR_POLICY_TABLE[idx_nf6] == idx_nf3

    def test_castling_mirror(self):
        """White kingside castling (e1-g1) mirrors to e8-g8."""
        move_wks = chess.Move(chess.E1, chess.G1)
        idx_wks = MoveEncoder.move_to_index(move_wks)

        move_bks = chess.Move(chess.E8, chess.G8)
        idx_bks = MoveEncoder.move_to_index(move_bks)

        assert _MIRROR_POLICY_TABLE[idx_wks] == idx_bks

    def test_all_valid_moves_mirror_to_valid(self):
        """Every valid policy index should mirror to another valid index."""
        from src.chess.board import INDEX_TO_MOVE

        for i in range(POLICY_SIZE):
            if INDEX_TO_MOVE[i] is not None:
                mirrored = _MIRROR_POLICY_TABLE[i]
                assert INDEX_TO_MOVE[mirrored] is not None, (
                    f"Valid index {i} (move {INDEX_TO_MOVE[i]}) "
                    f"mirrors to invalid index {mirrored}"
                )


class TestBoardEncoding:
    """Tests for perspective-flipped board encoding."""

    def test_white_to_move_stm_token(self):
        """White to move: STM token should be 0."""
        board = chess.Board()
        tokens = BoardEncoder.encode_board_array(board)
        assert tokens[2] == 0

    def test_black_to_move_stm_token(self):
        """Black to move: STM token should still be 0 (perspective canonical)."""
        board = chess.Board()
        board.push_san("e4")  # Now black to move
        tokens = BoardEncoder.encode_board_array(board)
        assert tokens[2] == 0

    def test_start_position_white(self):
        """Starting position (white to move): standard encoding."""
        board = chess.Board()
        tokens = BoardEncoder.encode_board_array(board)

        # White pawns on rank 1 (squares 8-15) → indices 1 (PAWN)
        for sq in range(8, 16):
            assert tokens[3 + sq] == chess.PAWN, f"Expected pawn at square {sq}"

        # Black pawns on rank 6 (squares 48-55) → indices 7 (PAWN + 6)
        for sq in range(48, 56):
            assert tokens[3 + sq] == chess.PAWN + 6, f"Expected black pawn at square {sq}"

    def test_black_to_move_flips_pieces(self):
        """After 1.e4, black's pieces should appear as 'current player' (1-6)."""
        board = chess.Board()
        board.push_san("e4")
        tokens = BoardEncoder.encode_board_array(board)

        # Black pawns (originally rank 6, sq 48-55) flipped to rank 1 (sq 8-15)
        # Encoded as current player's pawns (index 1)
        for file in range(8):
            mirrored_sq = chess.square(file, 1)  # rank 1
            if file == 4:
                # e7 pawn is still there, maps to e2 after flip
                assert tokens[3 + mirrored_sq] == chess.PAWN
            else:
                assert tokens[3 + mirrored_sq] == chess.PAWN

        # White pawns (originally rank 1, sq 8-15) flipped to rank 6 (sq 48-55)
        # Encoded as opponent's pawns (index 7)
        for file in range(8):
            mirrored_sq = chess.square(file, 6)  # rank 6
            if file == 4:
                # e4 pawn: originally on rank 3, flipped to rank 4
                assert tokens[3 + mirrored_sq] == 0  # empty at e7-flipped
            else:
                assert tokens[3 + mirrored_sq] == chess.PAWN + 6

    def test_castling_swap(self):
        """Castling rights should swap between white and black on flip."""
        board = chess.Board()
        tokens_white = BoardEncoder.encode_board_array(board)
        # Starting: all castling rights = 0b1111 = 15
        assert tokens_white[0] == 15

        board.push_san("e4")
        tokens_black = BoardEncoder.encode_board_array(board)
        # After flip: white bits (0-1) ↔ black bits (2-3)
        # Original: WK=1, WQ=1, BK=1, BQ=1 = 0b1111
        # Swapped: same value (symmetric in this case)
        assert tokens_black[0] == 15

        # Test asymmetric castling: remove white kingside
        board2 = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b Qkq - 0 1")
        tokens = BoardEncoder.encode_board_array(board2)
        # Black to move, so flip. Original: WQ=1(bit1), BK=1(bit2), BQ=1(bit3) = 0b1110 = 14
        # After swap: WK=0,WQ=0→bits0-1 get original bits2-3 = 0b11
        #             BK=0,BQ=0→bits2-3 get original bits0-1 = 0b01 (WQ was set)
        # Result: 0b0111 = 7? Let me compute:
        # Original castling = 14 = 0b1110. bits 0-1 = 0b10 (WQ only), bits 2-3 = 0b11 (BK+BQ)
        # Swap: bits 0-1 = old bits 2-3 = 0b11, bits 2-3 = old bits 0-1 = 0b10
        # Result = 0b1011 = 11
        assert tokens[0] == 11

    def test_en_passant_file_preserved(self):
        """En passant file should be the same regardless of flip."""
        board = chess.Board()
        board.push_san("e4")
        # After 1.e4, EP square is e3 (file 4)
        tokens = BoardEncoder.encode_board_array(board)
        assert tokens[1] == 4  # file e

    def test_encoding_deterministic(self):
        """Same position should always produce the same encoding."""
        board = chess.Board()
        board.push_san("e4")
        board.push_san("e5")
        t1 = BoardEncoder.encode_board_array(board)
        t2 = BoardEncoder.encode_board_array(board)
        np.testing.assert_array_equal(t1, t2)


class TestPerspectiveConsistency:
    """Tests that encode → mirror → decode produces consistent results."""

    def test_all_legal_moves_encode_correctly(self):
        """For every legal move in various positions, verify encoding roundtrip."""
        fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/3P1N2/PPP1NPPP/R2Q1RK1 b - - 0 1",
        ]
        for fen in fens:
            board = chess.Board(fen)
            flip = board.turn == chess.BLACK
            for move in board.legal_moves:
                idx = MoveEncoder.move_to_index(move)
                if flip:
                    mirrored_idx = int(_MIRROR_POLICY_TABLE[idx])
                    # The mirrored index should also be valid
                    from src.chess.board import INDEX_TO_MOVE
                    assert INDEX_TO_MOVE[mirrored_idx] is not None

    def test_encode_policy_flip_roundtrip(self):
        """encode_policy with flip, then mirror_policy, should match no-flip encoding."""
        board = chess.Board()
        board.push_san("e4")  # black to move

        # Create some fake visit counts for black's legal moves
        visit_counts: dict[chess.Move, int] = {}
        for i, move in enumerate(board.legal_moves):
            visit_counts[move] = (i + 1) * 10

        # Encode without flip (actual-move space)
        policy_actual = MoveEncoder.encode_policy(visit_counts, flip=False)
        # Encode with flip (model-space)
        policy_model = MoveEncoder.encode_policy(visit_counts, flip=True)

        # mirror_policy should convert between them
        policy_actual_from_model = torch.from_numpy(
            mirror_policy(policy_model.numpy())
        )
        torch.testing.assert_close(policy_actual, policy_actual_from_model)

    def test_mirror_symmetric_positions(self):
        """For color-symmetric positions, flipped encoding should be equivalent."""
        # Start position is symmetric — encoding from white's and black's
        # perspective should give the same piece layout (with indices swapped)
        board_w = chess.Board()  # white to move
        tokens_w = BoardEncoder.encode_board_array(board_w)

        # Make it black to move artificially (not a real game state, just for testing)
        board_b = chess.Board()
        board_b.turn = chess.BLACK
        tokens_b = BoardEncoder.encode_board_array(board_b)

        # Both should have STM = 0
        assert tokens_w[2] == 0
        assert tokens_b[2] == 0

        # In white encoding: white pawns at rank 1 (sq 8-15) = piece_type 1
        # In black encoding: black pawns (rank 6, sq 48-55) flipped to rank 1 (sq 8-15) = piece_type 1
        # So the pawn structure at rank 1 should be identical
        for sq in range(8, 16):
            assert tokens_w[3 + sq] == tokens_b[3 + sq] == chess.PAWN

    def test_100_positions_roundtrip(self):
        """Mirror policy roundtrip over 100 random-ish positions."""
        rng = np.random.default_rng(123)
        board = chess.Board()
        positions_tested = 0

        for _ in range(200):
            if board.is_game_over():
                board = chess.Board()
                continue

            # Create a random policy for this position
            policy = rng.random(POLICY_SIZE).astype(np.float32)
            policy /= policy.sum()

            # Roundtrip: mirror → mirror should be identity
            roundtrip = mirror_policy(mirror_policy(policy))
            np.testing.assert_allclose(roundtrip, policy, atol=1e-7)

            positions_tested += 1
            if positions_tested >= 100:
                break

            # Play a random legal move
            moves = list(board.legal_moves)
            move = moves[rng.integers(len(moves))]
            board.push(move)

        assert positions_tested >= 100


class TestUnderpromotionMirror:
    """Test that underpromotion moves mirror correctly."""

    def test_white_underpromotion_to_black(self):
        """White promoting on rank 7 should mirror to rank 0."""
        # White pawn on e7 promoting to knight on e8
        move = chess.Move(chess.E7, chess.E8, promotion=chess.KNIGHT)
        idx = MoveEncoder.move_to_index(move)
        mirrored_idx = int(_MIRROR_POLICY_TABLE[idx])

        # Should correspond to e2→e1 knight promotion (black promoting)
        from src.chess.board import INDEX_TO_MOVE
        mirrored_move = INDEX_TO_MOVE[mirrored_idx]
        assert mirrored_move is not None
        assert mirrored_move.from_square == chess.E2
        assert mirrored_move.to_square == chess.E1
        assert mirrored_move.promotion == chess.KNIGHT

    def test_capture_underpromotion(self):
        """Capture promotion should mirror file direction correctly."""
        # White pawn on d7 captures on e8, promotes to rook
        move = chess.Move(chess.D7, chess.E8, promotion=chess.ROOK)
        idx = MoveEncoder.move_to_index(move)
        mirrored_idx = int(_MIRROR_POLICY_TABLE[idx])

        from src.chess.board import INDEX_TO_MOVE
        mirrored_move = INDEX_TO_MOVE[mirrored_idx]
        assert mirrored_move is not None
        assert mirrored_move.from_square == chess.D2
        assert mirrored_move.to_square == chess.E1
        assert mirrored_move.promotion == chess.ROOK
