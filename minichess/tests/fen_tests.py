from minichess.chess.fastchess import Chess

from minichess.rl.chess_helpers import get_initial_chess_object
from random import choice

import chess

def standard_starting_pos():
    chess = get_initial_chess_object("8x8standard")
    fen = chess.fen()
    print(fen)
    assert fen == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def silverman_starting_pos():
    chess = get_initial_chess_object("5x4silverman")
    fen = chess.fen()
    print(fen)
    assert fen == "rqkr/pppp/4/PPPP/RQKR w - - 0 1"


def play_game():
    pychess_game = chess.Board()
    chess_game = get_initial_chess_object("8x8standard")
    while not pychess_game.is_game_over():
        move_to_make = chess.Move.from_uci(str(choice(list(pychess_game.legal_moves))))
        from_tuple = 7 - chess.square_rank(move_to_make.from_square), chess.square_file(move_to_make.from_square)
        to_tuple = 7 - chess.square_rank(move_to_make.to_square), chess.square_file(move_to_make.to_square)
        dx, dy = (to_tuple[0] - from_tuple[0], to_tuple[1] - from_tuple[1])
        promotion = -1 if move_to_make.promotion is None else move_to_make.promotion - 1
        chess_game.make_move(*from_tuple, dx, dy, promotion)
        pychess_game.push(move_to_make)
        pychess_fen, chess_fen = pychess_game.fen(en_passant="fen"), chess_game.fen()
        assert pychess_fen == chess_fen

def play_games(n=20):
    [play_game() for _ in range(n)]



if __name__ == "__main__":
    play_games(500)