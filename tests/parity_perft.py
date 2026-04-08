"""Perft-style parity check: compare legal moves at every tree node."""
import chess
from src.chess.bitboard import Board as BitboardBoard


def perft_compare(fen: str, depth: int, label: str) -> None:
    mismatches = []

    def walk(py_board: chess.Board, bb_board: BitboardBoard, d: int) -> None:
        if d == 0:
            return
        py_moves = list(py_board.legal_moves)
        bb_moves_set = {m.uci() for m in bb_board.legal_moves}
        py_moves_set = {m.uci() for m in py_moves}
        if py_moves_set != bb_moves_set:
            only_py = py_moves_set - bb_moves_set
            only_bb = bb_moves_set - py_moves_set
            mismatches.append({
                "fen": py_board.fen(),
                "only_py": sorted(only_py),
                "only_bb": sorted(only_bb),
            })
        for mv in py_moves:
            py_board.push(mv)
            bb_board.push(mv)
            walk(py_board, bb_board, d - 1)
            py_board.pop()
            bb_board.pop()

    py = chess.Board(fen)
    bb = BitboardBoard.from_fen(fen)
    walk(py, bb, depth)

    if mismatches:
        print("MISMATCH [" + label + "]: " + str(len(mismatches)) + " positions differ")
        for m in mismatches[:5]:
            print("  FEN: " + m["fen"])
            if m["only_py"]:
                print("    only in python-chess: " + str(m["only_py"]))
            if m["only_bb"]:
                print("    only in bitboard: " + str(m["only_bb"]))
    else:
        print("OK [" + label + "]")


if __name__ == "__main__":
    print("=== Perft-style game tree walks ===")

    perft_compare(chess.Board().fen(), 2, "startpos depth 2")

    # All white pawns on 7th, all black pawns on 2nd
    perft_compare(
        "8/PPPPPPPP/8/8/8/8/pppppppp/K5Bk w - - 0 1",
        2,
        "all pawns on 7th depth 2",
    )

    perft_compare(
        "8/4P1k1/8/8/8/8/8/4K3 w - - 0 1",
        2,
        "promotion check position depth 2",
    )

    perft_compare(
        "8/8/8/8/1q6/8/2P5/3K4 w - - 0 1",
        2,
        "pin position depth 2",
    )

    perft_compare(
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
        2,
        "castling position depth 2",
    )

    perft_compare(
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        2,
        "EP position depth 2",
    )

    perft_compare(
        "4r3/8/8/8/1b6/8/8/4K2k w - - 0 1",
        2,
        "double check depth 2",
    )

    perft_compare(
        "1n6/P7/8/8/8/8/8/K6k w - - 0 1",
        3,
        "capture promo depth 3",
    )

    perft_compare(
        "8/8/8/4k3/3K4/8/8/8 w - - 0 1",
        2,
        "kings adjacent depth 2",
    )

    # Standard starting position depth 3 (takes longer but more thorough)
    perft_compare(chess.Board().fen(), 3, "startpos depth 3")

    # Promotion stalemate
    perft_compare(
        "k7/1P6/1K6/8/8/8/8/8 w - - 0 1",
        2,
        "promotion stalemate depth 2",
    )

    # Kiwipete - famous perft position with many edge cases
    perft_compare(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        2,
        "kiwipete depth 2",
    )

    print("Done.")
