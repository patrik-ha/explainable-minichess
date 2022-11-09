from minichess.chess.fastchess import Chess
from minichess.chess.fastchess_utils import B_0, B_1, castling_attack_mask, castling_masks, diagonal_line_moves, flat, has_bit, inv_color, king_moves, knight_moves, load_board, pawn_attacks, pawn_moves_double, pawn_moves_single, print_bitboard, promotion_masks, set_bit, straight_line_moves, true_bits, unflat, unset_bit, visualize_board
import numpy as np
import json
import random
import string
import os
from typing import Tuple

from minichess.chess.magic import find_magic_bitboards, save_magic_bitboards


def random_string(n):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=n))


def launch_tensorboard(directory):
    from tensorboard import program
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", directory])
    url = tb.launch()
    print(f"Tensorboard listening on {url}")


def get_settings(config_file_path: str):
    with open(config_file_path) as f:
        return json.load(f)


def get_initial_chess_object(full_name: str):
    """Takes the name of the variant to play (e.g. 8x8standard) and returns a fully initialized Chess-object."""

    board_path = "minichess/boards/{}".format(full_name)
    bitboards, piece_lookup, dims = load_board(board_path)

    if not os.path.exists("minichess/chess/magics/{}x{}/diagonals.npz".format(*dims)):
        print("Need to calculate magics for this board dimension variant.")
        print("This only needs to be done once, but it can take a couple of minutes for large boards.")
        save_magic_bitboards(dims)

    data = np.load("minichess/chess/magics/{}x{}/diagonals.npz".format(*dims))
    diagonal_hash_table, diagonal_magics, diag_shift = data["hash_table"], data["magics"], data["shift"]

    data = np.load("minichess/chess/magics/{}x{}/straights.npz".format(*dims))
    straight_hash_table, straight_magics, straight_shift = data["hash_table"], data["magics"], data["shift"]
    PAWN_MOVES_SINGLE = pawn_moves_single(dims)
    PAWN_MOVES_DOUBLE = pawn_moves_double(dims)
    PAWN_ATTACKS = pawn_attacks(dims)
    KNIGHT_MOVES = knight_moves(dims)
    KING_MOVES = king_moves(dims)
    DIAGONAL_MOVES = diagonal_line_moves(dims)
    STRAIGHT_MOVES = straight_line_moves(dims)
    CASTLING_EMPTY_MASKS, CASTLING_ATTACK_MASKS, CASTLING_RIGHTS = castling_masks(dims, board_path)
    PROMOTION_MASKS = promotion_masks(dims)
    return Chess(
        bitboards.copy(),
        piece_lookup.copy(),
        dims,
        diagonal_hash_table,
        diagonal_magics,
        diag_shift,
        straight_hash_table,
        straight_magics,
        straight_shift,
        PAWN_MOVES_SINGLE,
        PAWN_MOVES_DOUBLE,
        PAWN_ATTACKS,
        KNIGHT_MOVES,
        KING_MOVES,
        DIAGONAL_MOVES,
        STRAIGHT_MOVES,
        CASTLING_EMPTY_MASKS,
        CASTLING_ATTACK_MASKS,
        PROMOTION_MASKS,
        CASTLING_RIGHTS
    )
