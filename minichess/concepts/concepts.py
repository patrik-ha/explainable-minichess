from minichess.chess.fastchess import Chess
from minichess.chess.fastchess_utils import inv_color, piece_matrix_to_legal_moves
import numpy as np


def in_check(chess: Chess):
    king_pos = chess.single_to_bitboard(*chess.find_king(chess.turn))
    return king_pos & chess.get_attacked_squares(inv_color(chess.turn), False)
    if chess.legal_move_cache is None:
        chess.legal_moves()
    return chess.any_checkers


def has_contested_open_file(chess: Chess):
    # First, find positions of own queens and rooks:
    squares_to_look_at = []
    for i in range(chess.dims[0]):
        for j in range(chess.dims[1]):
            piece_at = chess.piece_at(i, j, chess.turn)
            if piece_at == 4 or piece_at == 5:
                squares_to_look_at.append((i, j))

    for (i, j) in squares_to_look_at:
        # If the file, i, only has enemy and own rooks and queens on it, it is open and contested
        enemy_occupant_found = False
        for k in range(chess.dims[0]):
            own_piece_at = chess.piece_at(k, j, chess.turn)
            # Own piece on this file that is not a rook or queen
            if own_piece_at != -1 and own_piece_at != 4 and own_piece_at != 5:
                break
            enemy_piece_at = chess.piece_at(k, j, inv_color(chess.turn))
            # Enemy occupant on the file, ok...
            if enemy_piece_at == 4 or enemy_piece_at == 5:
                enemy_occupant_found = True
            # Some other enemy piece on the file
            if enemy_piece_at != 4 and enemy_piece_at != 5 and enemy_piece_at != -1:
                break
        else:
            # This means the file is open, and if it has an enemy oppucant as well, it is contested
            if enemy_occupant_found:
                return True
    return False


def opponent_has_mate_threat(chess: Chess):
    # Essentially, pass the turn over, and see if enemy (now player to move) has mate
    to_check = chess.copy()
    to_check.make_null_move()
    return has_mate_threat(to_check)


def has_mate_threat(chess: Chess):
    moves, proms = chess.legal_moves()
    legal_moves = piece_matrix_to_legal_moves(moves, proms)
    for move in legal_moves:
        (i, j), (dx, dy), prom = move
        potential_mate = chess.copy()
        potential_mate.make_move(i, j, dx, dy, prom)
        if potential_mate.game_result() is not None and abs(potential_mate.game_result()) == 1:
            return True
    return False


def threat_opp_queen(chess: Chess):
    if chess.legal_move_cache is None:
        chess.legal_moves()

    enemy_turn = inv_color(chess.turn)
    if chess.bitboards[enemy_turn, 4] == 0:
        return False

    queen_pos = chess.find_queen(enemy_turn)

    moves, proms = chess.legal_moves()
    legal_moves = piece_matrix_to_legal_moves(moves, proms)
    for move in legal_moves:
        (i, j), (dx, dy), prom = move
        if i + dx == queen_pos[0] and j + dy == queen_pos[1]:
            return True

    return False


def material_advantage(position: Chess):
    total = sum_of_pieces(position, position.turn) - sum_of_pieces(position, inv_color(position.turn))
    return total >= 3


def random(position: Chess):
    return np.random.random() > 0.5


def sum_of_pieces(position: Chess, color: bool):
    piece_values = {
        0: 1,
        1: 2,
        2: 2,
        3: 4,
        4: 9,
        5: 0
    }
    total = 0
    for i in range(position.dims[0]):
        for j in range(position.dims[1]):
            if position.piece_lookup[color, i, j] == -1:
                continue
            total += piece_values[position.piece_lookup[color, i, j]]

    return total
