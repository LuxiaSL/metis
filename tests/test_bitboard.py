import random
import unittest

import chess
import numpy as np

from src.chess.bitboard import Board as BitboardBoard
from src.chess.board import BoardEncoder


def _perft(board: BitboardBoard, depth: int) -> int:
    if depth == 0:
        return 1

    total = 0
    for move in board.legal_move_codes():
        board.push(int(move))
        total += _perft(board, depth - 1)
        board.pop()
    return total


class TestBitboardBoard(unittest.TestCase):
    def _assert_position_matches(
        self,
        py_board: chess.Board,
        bb_board: BitboardBoard,
        *,
        claim_draw: bool = True,
    ) -> None:
        py_moves = sorted(move.uci() for move in py_board.legal_moves)
        bb_moves = sorted(move.uci() for move in bb_board.legal_moves)
        self.assertEqual(py_moves, bb_moves, py_board.fen(en_passant="fen"))

        self.assertEqual(
            py_board.is_game_over(claim_draw=False),
            bb_board.is_game_over(claim_draw=False),
            py_board.fen(en_passant="fen"),
        )
        self.assertEqual(
            py_board.is_game_over(claim_draw=claim_draw),
            bb_board.is_game_over(claim_draw=claim_draw),
            py_board.fen(en_passant="fen"),
        )
        self.assertEqual(
            py_board.result(claim_draw=False),
            bb_board.result(claim_draw=False),
            py_board.fen(en_passant="fen"),
        )
        self.assertEqual(
            py_board.result(claim_draw=claim_draw),
            bb_board.result(claim_draw=claim_draw),
            py_board.fen(en_passant="fen"),
        )

        np.testing.assert_array_equal(
            BoardEncoder.encode_board_array(py_board),
            BoardEncoder.encode_board_array(bb_board),
            err_msg=py_board.fen(en_passant="fen"),
        )

    def test_perft_start_and_kiwipete(self) -> None:
        start = BitboardBoard()
        self.assertEqual([_perft(start, d) for d in range(1, 5)], [20, 400, 8902, 197281])

        kiwipete = BitboardBoard.from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        self.assertEqual([_perft(kiwipete, d) for d in range(1, 4)], [48, 2039, 97862])

    def test_random_game_comparison(self) -> None:
        rng = random.Random(0)
        positions_checked = 0

        for _game in range(20):
            py_board = chess.Board()
            bb_board = BitboardBoard()

            for _ply in range(80):
                self._assert_position_matches(py_board, bb_board)
                positions_checked += 1

                if py_board.is_game_over(claim_draw=True):
                    break

                move = rng.choice(list(py_board.legal_moves))
                py_board.push(move)
                bb_board.push(move)

        self.assertGreaterEqual(positions_checked, 1000)

    def test_edge_case_positions_match_python_chess(self) -> None:
        fens = [
            "r3k2r/8/8/8/8/8/5r2/R3K2R w KQkq - 0 1",
            "r3k2r/8/8/8/8/8/4r3/R3K2R w KQkq - 0 1",
            "2b1k1nr/4q3/2p2r1p/1P1pPp1P/p7/P1bP4/3B1QRN/RN2KB2 w k d6 0 26",
            "k7/P7/8/8/8/8/8/7K w - - 0 1",
            "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
            "8/8/8/8/8/8/7B/K6k w - - 0 1",
        ]

        for fen in fens:
            with self.subTest(fen=fen):
                py_board = chess.Board(fen)
                bb_board = BitboardBoard.from_fen(fen)
                self._assert_position_matches(py_board, bb_board)

    def test_repetition_and_fifty_move_claims(self) -> None:
        py_board = chess.Board()
        bb_board = BitboardBoard()

        repetition_cycle = ["g1f3", "g8f6", "f3g1", "f6g8"] * 2
        for uci in repetition_cycle:
            move = chess.Move.from_uci(uci)
            py_board.push(move)
            bb_board.push(move)

        self._assert_position_matches(py_board, bb_board)
        self.assertFalse(py_board.is_game_over(claim_draw=False))
        self.assertTrue(py_board.is_game_over(claim_draw=True))

        fifty_fen = "8/8/8/8/8/8/8/KR5k w - - 100 1"
        py_fifty = chess.Board(fifty_fen)
        bb_fifty = BitboardBoard.from_fen(fifty_fen)
        self._assert_position_matches(py_fifty, bb_fifty)
        self.assertFalse(bb_fifty.is_game_over(claim_draw=False))
        self.assertTrue(bb_fifty.is_game_over(claim_draw=True))


if __name__ == "__main__":
    unittest.main()
