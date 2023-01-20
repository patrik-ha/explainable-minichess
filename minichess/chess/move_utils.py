from functools import lru_cache

from numba import jit, njit, prange
import numpy as np


@jit
def inv_color(color):
    return 1 - color


def calculate_all_moves(dims):
    rook_moves = []
    for i in range(dims[0]):
        if i == 0:
            continue
        rook_moves.append((i, 0))
        rook_moves.append((-i, 0))
    for i in range(dims[1]):
        if i == 0:
            continue
        rook_moves.append((0, i))
        rook_moves.append((0, -i))

    bishop_moves = []
    for i in range(min(dims)):
        if i == 0:
            continue
        bishop_moves.append((i, i))
        bishop_moves.append((-i, i))
        bishop_moves.append((i, -i))
        bishop_moves.append((-i, -i))
    knight_moves = []

    knight_delta = np.array([1, 2, -1, -2])
    for i in knight_delta:
        for j in knight_delta:
            if abs(i) + abs(j) != 3:
                continue
            knight_moves.append((i, j))

    all_moves = list(set(rook_moves + bishop_moves + knight_moves))

    total = len(all_moves)
    promotions = np.array([1, 2, 3, 4])

    promotions_dy = [-1, 0, 1]
    # Max 8x8 with max 5 promotions (first is no promotion)

    # By using int16 and int8 here, it is assumed that...
    # The absolute value of dx, dy, or the number of unique promotions is larger than 128
    # And that the total amount of moves per square is no larger than 25565
    all_moves_dict = np.full((16, 16, max(promotions) + 1), -2000, dtype=np.int16)
    all_moves_inv = np.zeros((total + (max(promotions) + 1) * len(promotions_dy), 3), dtype=np.int8)
    # index: [dx, dy, prom]
    for (i, (dx, dy)) in enumerate(all_moves):
        all_moves_dict[dx + 8, dy + 8, 0] = i + 1
        all_moves_inv[i] = np.array([dx, dy, 0], dtype=np.int8)

    total = len(all_moves)
    for promotion in promotions:
        for dy in promotions_dy:
            all_moves_dict[-1 + 8, dy + 8, promotion] = total + 1
            all_moves_inv[total] = np.array([-1, dy, promotion], dtype=np.int8)
            total += 1
    return all_moves_dict, all_moves_inv


def move_to_index(all_moves, dx: int, dy: int, promotion: int, color: bool):
    if color == 0:
        dx *= -1
        dy *= -1
    if promotion == -1:
        promotion = 0
    return all_moves[dx + 8, dy + 8, promotion] - 1

@njit
def index_to_move(all_moves_inv, index: int, color: bool):
    dx, dy, promotion = all_moves_inv[index]
    if color == 0:
        dx *= -1
        dy *= -1
    if promotion == 0:
        promotion = -1
    return dx, dy, promotion


def flat_move_to_partial(all_moves_inv, dims, flat_move_index, color: bool):
    i, j, ind = np.unravel_index(flat_move_index, (dims[0], dims[1], all_moves_inv.shape[0]))
    dx, dy, promotion = all_moves_inv[ind]
    if color == 0:
        dx *= -1
        dy *= -1
    if promotion == 0:
        promotion = -1
    return i, j, dx, dy, promotion


@njit(cache=True)
def in_bounds(i, j, dims):
    return i >= 0 and i < dims[0] and j >= 0 and j < dims[1]


@njit(cache=True)
def in_bounds_x(i, dims):
    return i >= 0 and i < dims[0]


@njit(cache=True)
def in_bounds_y(j, dims):
    return j >= 0 and j < dims[1]


@njit(cache=True)
def knight_moves(i, j, dims):
    moves = []
    for dx in [1, 2, -1, -2]:
        for dy in [1, 2, -1, -2]:
            if abs(dx) + abs(dy) != 3:
                continue

            if in_bounds(i + dx, j + dy, dims):
                moves.append((dx, dy))
    return moves


@njit(cache=True)
def king_moves(i, j, dims):
    moves = []
    for dx in [1, 0, -1]:
        for dy in [1, 0, -1]:
            if dx == 0 and dy == 0:
                continue
            if in_bounds(i + dx, j + dy, dims):
                moves.append((dx, dy))
    return moves


@njit
def find_king(board, turn):
    flat = board[turn, :, :, 5].argmax()
    return flat // board.shape[1], flat % board.shape[2]


if __name__ == "__main__":
    result = calculate_all_moves(np.array([4, 5]))
