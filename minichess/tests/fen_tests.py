from minichess.chess.fastchess import Chess

from minichess.rl.chess_helpers import get_initial_chess_object

def standard_starting_pos():
    chess = get_initial_chess_object("8x8standard")

    assert chess.fen() == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

if __name__ == "__main__":
    standard_starting_pos()