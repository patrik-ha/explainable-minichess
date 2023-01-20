import chess as pychess
from functools import lru_cache
import os
import numpy as np
from numba import jit, njit


@jit
def inv_color(color):
    return 1 - color


PIECE_LOOKUP = {
    'p': 0,
    'r': 3,
    'n': 1,
    'b': 2,
    'k': 5,
    'q': 4
}

PAWN = 0,
KNIGHT = 1,
BISHOP = 2,
ROOK = 3,
QUEEN = 4,
KING = 5


WHITE = 1
BLACK = 0

B_1 = np.uint64(1)
B_0 = np.uint64(0)

INVERSE_PIECE_LOOKUP = {v: k for k, v in PIECE_LOOKUP.items()}


def load_board(board_setup_path="minichess/boards/8x8standard"):
    with open(board_setup_path + ".board") as f:
        lines = [line.strip() for line in f.readlines()]

    dimensions = (len(lines), len(lines[0]))

    bitboards = np.zeros((2, 6), dtype=np.uint64)

    piece_lookup = np.full((2, *dimensions), -1, dtype=np.int8)

    for i, line in enumerate(lines):
        for j, c in enumerate(line):
            f = flat(i, j, dimensions)

            if c == " " or c == ".":
                continue
            piece_type = PIECE_LOOKUP[c.lower()]
            color = WHITE if c.upper() == c else BLACK
            piece_lookup[color, i, j] = piece_type
            bitboards[color, piece_type] = set_bit(bitboards[color, piece_type], f)

    return bitboards, piece_lookup, dimensions


def flat(i, j, dims):
    # Flattens a (i, j)-tuple into a flattened index
    # Also takes "dims" as a deprecated parameter. TODO: remove this
    return np.uint64(8 * i + j)

@njit
def unflat(f, dims):
    # Bad name, but translates a flattened index to its corresponding (i, j)-coordinate tuple
    return int(f // 8), int(f % 8)


def set_bit(bitboard, bit):
    return np.uint64(bitboard) | (B_1 << bit)


def unset_bit(bitboard, bit):
    return np.uint64(bitboard) & ~(B_1 << bit)


@jit(nopython=True)
def has_bit(bitboard, bit):
    return (bitboard >> np.uint64(bit)) & B_1


def in_bounds(i, j, dims):
    return i >= 0 and i < dims[0] and j >= 0 and j < dims[1]


def knight_moves(dims):
    """Make a bitboard describing all knight attacks from (i, j) to other squares."""
    knight_moves = np.zeros(dims, dtype=np.uint64)
    for i in range(dims[0]):
        for j in range(dims[1]):
            f = flat(i, j, dims)
            for dx in [1, 2, -1, -2]:
                for dy in [1, 2, -1, -2]:
                    if abs(dx) + abs(dy) != 3:
                        continue

                    if in_bounds(i + dx, j + dy, dims):
                        f = flat(i + dx, j + dy, dims)

                        # Set the bit corresponding to (i+dx, j+dy) for the position (i, j)
                        # Meaning that it is possible to move (dx, dy) from the position (i, j)
                        knight_moves[i, j] |= B_1 << f
    return knight_moves


def print_bitboard(bitboard, dimensions):
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            f = dimensions[1] * i + j
            print(1 if has_bit(bitboard, f) else 0, end=" ")
        print()


def king_moves(dims):
    king_moves = np.zeros(dims, dtype=np.uint64)
    for i in range(dims[0]):
        for j in range(dims[1]):
            for dx in [1, 0, -1]:
                for dy in [1, 0, -1]:
                    if dx == 0 and dy == 0:
                        continue
                    if in_bounds(i + dx, j + dy, dims):
                        f = flat(i + dx, j + dy, dims)
                        king_moves[i, j] |= B_1 << f
    return king_moves


def pawn_attacks(dims):
    pawn_attacks = np.zeros((2, *dims), dtype=np.uint64)
    for turn in [0, 1]:
        row_inc = -1 if turn else 1
        for i in range(0, dims[0]):
            if i + row_inc >= 0 and i + row_inc < dims[0]:
                for j in range(dims[1]):
                    if j != 0:
                        pawn_attacks[turn, i, j] |= B_1 << flat(i + row_inc, j - 1, dims)
                    if j != dims[1] - 1:
                        pawn_attacks[turn, i, j] |= B_1 << flat(i + row_inc, j + 1, dims)
    return pawn_attacks


def pawn_moves_single(dims):
    pawn_moves = np.zeros((2, *dims), dtype=np.uint64)
    for turn in [0, 1]:
        row_inc = -1 if turn else 1
        start_rank = dims[0] - 2 if turn else 1
        top = 0 if turn else dims[0] - 1
        for i in range(1, dims[0] - 1):
            for j in range(dims[1]):
                if i != top:
                    pawn_moves[turn, i, j] |= B_1 << flat(i + row_inc, j, dims)
    return pawn_moves


def pawn_moves_double(dims):
    pawn_moves = np.zeros((2, *dims), dtype=np.uint64)
    for turn in [0, 1]:
        row_inc = -1 if turn else 1
        start_rank = dims[0] - 2 if turn else 1
        top = 0 if turn else dims[0] - 1
        for i in range(1, dims[0] - 1):
            for j in range(dims[1]):
                if i == start_rank:
                    pawn_moves[turn, i, j] |= B_1 << flat(i + 2 * row_inc, j, dims)
    return pawn_moves


def diagonal_line_moves(dims):
    diagonal_moves = np.zeros(dims, dtype=np.uint64)
    for i in range(dims[0]):
        for j in range(dims[1]):
            for sign_x in [-1, 1]:
                for sign_y in [-1, 1]:
                    for di in range(1, max(dims)):
                        # Her er det stopp, av brettet
                        if not in_bounds(i + sign_x * di, j + sign_y * di, dims):
                            break

                        f = flat(i + sign_x * di, j + sign_y * di, dims)

                        diagonal_moves[i, j] |= B_1 << f
    return diagonal_moves


def straight_line_moves(dims):
    straight_moves = np.zeros(dims, dtype=np.uint64)
    for i in range(dims[0]):
        for j in range(dims[1]):
            for sign_x, sign_y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                for di in range(1, max(dims)):
                    # Her er det stopp, av brettet
                    if not in_bounds(i + sign_x * di, j + sign_y * di, dims):
                        break

                    f = flat(i + sign_x * di, j + sign_y * di, dims)

                    straight_moves[i, j] |= B_1 << f
    return straight_moves


def castling_attack_mask(dims):
    """Should return a mask that specifies the squares that have to not be under attack for castling to occur. Per color per side."""
    if dims[1] != 8:
        # Could define something more here
        return B_0

    # Color, side
    masks = np.zeros((2, 2), dtype=np.uint64)
    for turn in [0, 1]:
        back_rank = 0 if not turn else dims[0] - 1
        # Ugly but it works
        masks[turn, 0] = set_bit(masks[turn, 0], flat(back_rank, 4, dims))
        masks[turn, 0] = set_bit(masks[turn, 0], flat(back_rank, 3, dims))
        masks[turn, 0] = set_bit(masks[turn, 0], flat(back_rank, 2, dims))
        masks[turn, 1] = set_bit(masks[turn, 1], flat(back_rank, 4, dims))
        masks[turn, 1] = set_bit(masks[turn, 1], flat(back_rank, 5, dims))
        masks[turn, 1] = set_bit(masks[turn, 1], flat(back_rank, 6, dims))

    return masks


def castling_masks(dims, board_name):
    """Should return masks that specifies the squares that have to be empty and not attacked for castling to occur. Per color per side."""
    empty_masks = np.zeros((2, 2), dtype=np.uint64)
    attack_masks = np.zeros((2, 2), dtype=np.uint64)
    castling_rights = np.zeros((2, 2), dtype=np.uint8)
    if not os.path.exists(board_name + ".castle"):
        return empty_masks, attack_masks, castling_rights
    castling_rights = np.ones((2, 2), dtype=np.uint8)
    with open(board_name + ".castle") as f:
        line_pairs = f.read().split("\n\n")

    for p, pair in enumerate(line_pairs):
        pair = pair.split("\n")
        pair = [p.strip() for p in pair]
        print(pair)
        empty_mask, attack_mask = pair

        for turn in [0, 1]:
            back_rank = 0 if not turn else dims[0] - 1
            for j in range(dims[1]):
                if empty_mask[j] == '1':
                    empty_masks[turn, p] = set_bit(empty_masks[turn, p], flat(back_rank, j, dims))
                if attack_mask[j] == '1':
                    attack_masks[turn, p] = set_bit(attack_masks[turn, p], flat(back_rank, j, dims))

    return empty_masks, attack_masks, castling_rights


def promotion_masks(dims):
    """Should return a mask that specifies that any pawn moves to the mask results in a promotion. Per color."""
    masks = np.zeros(2, dtype=np.uint64)
    for j in range(dims[1]):
        masks[0] = set_bit(masks[0], flat(dims[0] - 1, j, dims))
        masks[1] = set_bit(masks[1], flat(0, j, dims))
    return masks

@njit
def agent_state(dims, bitboards, castling_rights, turn, en_passant, has_en_passant, ply_count_without_adv):
    full_state = np.zeros((dims[0], dims[1], 4 + 3 + 2 * 6), dtype=np.float32)

    # Brett for trekk og motstander
    for turn in [0, 1]:
        for piece_type in range(6):
            for bit in true_bits(bitboards[turn, piece_type]):
                i, j = unflat(bit, dims)
                full_state[i, j, 6 * turn + piece_type] = 1
    offset = 6 * turn + 6
    # Mine rokeringsmuligheter
    full_state[:, :, offset] = castling_rights[turn, 0]
    full_state[:, :, offset + 1] = castling_rights[turn, 1]
    # Deres rokeringsmuligheter
    full_state[:, :, offset + 2][:, :] = castling_rights[inv_color(turn), 0]
    full_state[:, :, offset + 3][:, :] = castling_rights[inv_color(turn), 1]

    en_passant_plane = np.zeros((dims[0], dims[1]), dtype=np.float32)
    if has_en_passant:
        en_passant_plane[en_passant[0], en_passant[1]] = 1

    full_state[:, :, offset + 4] = en_passant_plane
    full_state[:, :, offset + 5][:, :] = ply_count_without_adv / 20
    full_state[:, :, offset + 6][:, :] = turn

    if turn == 0:
        full_state = np.fliplr(full_state)
        full_state = np.flipud(full_state)
    return full_state

@njit
def true_bits(num):
    while num:
        temp = num & -num
        num -= temp
        yield int(np.log2(temp))

@njit
def piece_matrix_to_legal_moves(matrix, promotions):
    moves = []
    valid_promotions = [1, 2, 3, 4]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == 0:
                continue
            for ind in true_bits(matrix[i, j]):
                target = unflat(ind, matrix.shape)
                origin = (i, j)
                deltas = (target[0] - i, target[1] - j)
                if has_bit(promotions[i, j], ind):
                    for prom in valid_promotions:
                        moves.append((origin, deltas, prom))
                else:
                    moves.append((origin, deltas, -1))

    return moves

@njit
def move_to_index(all_moves, dx: int, dy: int, promotion: int, color: bool):
    if color == 0:
        dx *= -1
        dy *= -1
    if promotion == -1:
        promotion = 0
    return all_moves[dx + 8, dy + 8, promotion] - 1

@njit
def legal_moves_to_illegal_move_mask(moves, proms, child_priors, all_moves, player_number):
    legal_moves = piece_matrix_to_legal_moves(moves, proms)
    illegal_moves_mask = np.ones(child_priors)
    for move in legal_moves:
        (i, j), (dx, dy), promotion = move
        ind = move_to_index(all_moves, dx, dy, promotion, player_number)
        illegal_moves_mask[i, j, ind] = 0
    return illegal_moves_mask

@njit
def child_Q(child_win_value, child_number_visits):
    return child_win_value / (1 + child_number_visits)

@njit
def child_U(cpuct, number_visits, child_priors, child_number_visits):
    return cpuct * np.sqrt(number_visits) * (
        child_priors / (1 + child_number_visits))

@njit
def get_best_child(player_number, child_win_value, child_number_visits, cpuct, number_visits, child_priors, illegal_moves_mask):
    if player_number == 0:
        return np.argmin(child_Q(child_win_value, child_number_visits) - child_U(cpuct, number_visits, child_priors, child_number_visits) + illegal_moves_mask * 100000)
    else:
        return np.argmax(child_Q(child_win_value, child_number_visits) + child_U(cpuct, number_visits, child_priors, child_number_visits) - illegal_moves_mask * 100000)

@njit
def prior_math(illegal_moves_mask, dims, child_priors, move_cap, cnoise, value_estimate, turn):
    if turn == 0:
        value_estimate *= -1
    child_priors = np.reshape(child_priors, (dims[0], dims[1], move_cap))
    noise = np.random.uniform(0.0, 1.0, size=dims[0] * dims[1] * move_cap)
    # Sum of noise is equal to 1
    noise = noise.reshape(child_priors.shape)

    noise = noise * (illegal_moves_mask == 0)

    noise /= noise.sum()

    child_priors = (1 - cnoise) * child_priors + cnoise * noise
    child_priors /= child_priors.sum()
    return child_priors

def visualize_board(bitboards, dims):
    for i in range(dims[0]):
        for j in range(dims[1]):
            f = flat(i, j, dims)
            symbol = "."
            for piece in range(6):
                if has_bit(bitboards[0, piece], f):
                    if symbol != ".":
                        symbol = "#"
                    else:
                        symbol = INVERSE_PIECE_LOOKUP[piece]
                if has_bit(bitboards[1, piece], f):
                    if symbol != ".":
                        symbol = "#"
                    else:
                        symbol = INVERSE_PIECE_LOOKUP[piece].upper()
            print(symbol + " ", end="")
        print()
    print()


def chess_move_to_uci(move, dims):
    """Takes a move from the chess-object, and translates it to UCI."""
    (i, j), (dx, dy), promotion = move
    return coordinate_to_square_name(i, j, dims) + coordinate_to_square_name(i + dx, j + dy, dims) + ("" if promotion == -1 else INVERSE_PIECE_LOOKUP[promotion])


def square_name_to_coordinate_move(coordinate, board_dimensions):
    rank = int(coordinate[1]) - 1
    file = ord(coordinate[0]) - ord('a')

    return board_dimensions[1] - 1 - rank, file


def uci_move_to_native_move(uci_move, board):
    origin, target = uci_move[:2], uci_move[2:4]

    i, j = square_name_to_coordinate_move(origin, board.dims)
    l, k = square_name_to_coordinate_move(target, board.dims)
    dx, dy = l - i, k - j
    promotion = -1
    if len(uci_move) > 4:
        promotion = PIECE_LOOKUP[uci_move[-1]]

    return i, j, dx, dy, promotion


def more_than_one_bit_set(board):
    return board & (board - B_1) != 0


def bit_count(board):
    return int(board).bit_count()


def coordinate_to_square_name(x, y, board_dims):
    return pychess.FILE_NAMES[y] + pychess.RANK_NAMES[board_dims[0] - 1 - x]
