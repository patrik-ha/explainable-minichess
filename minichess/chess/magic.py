import collections
import os
from queue import Queue
from .fastchess_utils import B_0, flat, has_bit, in_bounds, print_bitboard, set_bit, straight_line_moves
from .fastchess_utils import diagonal_line_moves
import itertools

import numpy as np


def find_magic_bitboard(i, j, dims, directions_to_look, all_possible_moves, magic_shift):
    """
    Finds an array 'moves' and a magic number y so that moves[(x * y) >> (64-magic_shift)] 
    are the available moves given the total amount of pieces on the board x,
    and z is a bitboard with the available moves from (i, j) in the current piece-situation.

    Essentially creates a look-up table mapping from x (the pieces on the board) to the available moves,
    given the hashing function (x * y) >> (64-magic_shift).
    """
    all_moves = all_possible_moves[i, j]
    # print_bitboard(all_moves, dims)

    origin = set_bit(0, flat(i, j, dims))

    bits_to_be_occupied = []
    for l in range(dims[0]):
        for m in range(dims[1]):
            f = flat(l, m, dims)
            if has_bit(all_moves, f):
                bits_to_be_occupied.append(f)

    questions = []
    answers = []
    # Find all possible
    for blockers in list(itertools.product([0, 1], repeat=len(bits_to_be_occupied))):
        altered = B_0
        for index, bit in enumerate(blockers):
            if bit:
                altered = set_bit(altered, bits_to_be_occupied[index])

        # This is the bitboard with the ACTUAL moves that can be made from the position,
        # given the pieces on the board designated by 'blockers'
        available_moves = find_connected_components(((~altered & all_moves) | origin), i, j, dims, directions_to_look) & ~origin
        questions.append((altered & all_moves))
        answers.append(available_moves)
    np.seterr(over="ignore")
    found = False
    collisions = 0
    while not found:
        hash_table = np.zeros(2**magic_shift, dtype=np.uint64)
        magic = np.random.randint(np.iinfo(np.uint64).max, dtype=np.uint64)
        for question, answer in zip(questions, answers):
            masked = (question * magic) >> np.uint64(64 - magic_shift)
            # Collision: This 'hash-value' already has an answer, but it's not the same one!
            if hash_table[masked] != 0 and hash_table[masked] != answer:
                collisions += 1
                if collisions > 20000:
                    raise TimeoutError("Too many collisions. Increment shift.")
                break
            hash_table[masked] = answer
        else:
            # print("collisions:", collisions)
            found = True

    return hash_table, magic


def find_connected_components(bitboard, i, j, dims, directions_to_look):
    """Uses BFS to find bits that are connected, sets these, essentially finding available diag moves for bishop."""
    queue = []

    queue.append((i, j))

    result = B_0

    visited = set()
    while len(queue):
        to_visit = queue.pop()

        if to_visit in visited:
            continue

        visited.add(to_visit)
        f = flat(to_visit[0], to_visit[1], dims)

        # This means that occupants are also included in the magic
        # This is because they can be enemy pieces, and so captured.
        # This is also useful for calculating rays
        result = set_bit(result, f)
        if has_bit(bitboard, f):

            for dx, dy in directions_to_look:
                n_i, n_j = to_visit[0] + dx, to_visit[1] + dy

                if in_bounds(to_visit[0] + dx, to_visit[1] + dy, dims):
                    queue.append((to_visit[0] + dx, to_visit[1] + dy))

    return result


def find_magic_bitboards(dims, directions_to_look, all_possible_moves, magic_shift):
    done_calculating = False
    while not done_calculating:
        try:
            magics = np.zeros(dims, dtype=np.uint64)
            hash_tables = np.zeros((*dims, 2**magic_shift), dtype=np.uint64)

            for i in range(dims[0]):
                for j in range(dims[1]):
                    hash_table, magic = find_magic_bitboard(i, j, dims, directions_to_look, all_possible_moves, magic_shift)
                    hash_tables[i, j] = hash_table
                    magics[i, j] = magic
            done_calculating = True
        except TimeoutError:
            magic_shift += 1
            print("Too many collisions. Incrementing shift to {}.".format(magic_shift))
    return hash_tables, magics, magic_shift


def find_magic_bitboards_for_diagonals(dims, shift):
    return find_magic_bitboards(dims, [(-1, 1), (1, -1), (-1, -1), (1, 1)], diagonal_line_moves(dims), shift)


def find_magic_bitboards_for_straights(dims, shift):
    return find_magic_bitboards(dims, [(-1, 0), (1, 0), (0, -1), (0, 1)], straight_line_moves(dims), shift)


def save_magic_bitboards(dims, minichess_path, shift=None):
    if shift is None:
        shift = magic_shift_start_estimate(dims)
        print("Starting estimate for shift: {}".format(shift))
    print("Starting calculations for diagonal magics.")
    os.makedirs("{}/chess/magics/{}x{}".format(minichess_path, *dims), exist_ok=True)
    diag_hash, diag_magics, diag_shift = find_magic_bitboards_for_diagonals(dims, shift)
    np.savez("{}/chess/magics/{}x{}/diagonals".format(minichess_path, *dims), hash_table=diag_hash, magics=diag_magics, shift=diag_shift)
    print("Diagonal magics done!")

    print("Starting calculations for straight magics.")
    straight_hash, straight_magics, straight_shift = find_magic_bitboards_for_straights(dims, shift)
    np.savez("{}/chess/magics/{}x{}/straights".format(minichess_path, *dims), hash_table=straight_hash, magics=straight_magics, shift=straight_shift)
    print("Straight line magics done!")


def magic_shift_start_estimate(dims):
    if dims[0] * dims[1] == 64:
        return 17
    if dims[0] * dims[1] == 36:
        return 11

    # "Approximately"
    # Know that for 8x8, 17 is ok, and for 6x6, 11 is ok, so this 'works'
    # For very small boards, the size is very small anyway.
    # It's just important that the shifts are as small as possible for large boards, since they quickly become very large.
    return int(0.21 * dims[0] * dims[1] + 4.29)


if __name__ == "__main__":
    save_magic_bitboards((8, 8))
