import random
from math import log, sqrt
import numpy as np
from minichess.chess.fastchess import Chess

from minichess.chess.fastchess_utils import piece_matrix_to_legal_moves
from minichess.chess.move_utils import index_to_move, move_to_index


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class Node:
    """
    A node in MCTS.
    Optimizations based on https://www.moderndescartes.com/essays/deep_dive_mcts/.
    """

    def __init__(self, state, move_indices, move_cap, all_moves, all_moves_inv, cpuct, nondet_plies, prior=0):
        self.state = state
        self.parent = None
        self.move_cap = move_cap
        self.children = {}
        self.expanded = False
        self.player_number = int(state.turn)
        self.all_moves = all_moves
        self.all_moves_inv = all_moves_inv
        self.move_indices = move_indices
        self.prior = 0
        self.cpuct = cpuct
        self.nondet_plies = nondet_plies

        self.child_priors = np.zeros((self.state.dims[0], self.state.dims[1], move_cap), dtype=np.float32)
        self.child_win_value = np.zeros((self.state.dims[0], self.state.dims[1], move_cap), dtype=np.float32)
        self.child_number_visits = np.zeros((self.state.dims[0], self.state.dims[1], move_cap), dtype=np.float32)

        moves, proms = self.state.legal_moves()
        legal_moves = piece_matrix_to_legal_moves(moves, proms)
        self.illegal_moves_mask = np.ones_like(self.child_priors)
        for move in legal_moves:
            (i, j), (dx, dy), promotion = move
            ind = move_to_index(self.all_moves, dx, dy, promotion, self.player_number)
            self.illegal_moves_mask[i, j, ind] = 0

    @property
    def number_visits(self):
        # Is root node, all visits go through it
        if self.parent is None:
            return np.sum(self.child_number_visits)
        return self.parent.child_number_visits[self.move_indices]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.move_indices] = value

    @property
    def win_value(self):
        return self.parent.child_win_value[self.move_indices]

    @win_value.setter
    def win_value(self, value):
        self.parent.child_win_value[self.move_indices] = value

    def select_leaf(self):
        current = self
        while current.expanded:
            child_move = current.get_best_child()
            # This is a (deferred) leaf, have to create it
            if child_move not in current.children:
                current.add_child(child_move, self.child_priors[child_move])
            current = current.children[child_move]
        return current

    def expand(self, child_priors):
        self.expanded = True
        self.child_priors = child_priors

    def add_child(self, move, prior):
        child_state = self.state.copy()
        i, j, ind = move
        dx, dy, promotion = index_to_move(self.all_moves_inv, ind, self.player_number)
        child_state.make_move(i, j, dx, dy, promotion)
        child = Node(child_state, (i, j, ind), self.move_cap, self.all_moves, self.all_moves_inv, self.cpuct, self.nondet_plies, prior)
        child.parent = self
        self.children[(i, j, ind)] = child

    def update_win_value(self, value):
        if self.parent:
            self.number_visits += 1
            self.win_value += value
            self.parent.update_win_value(value)

    def child_Q(self):
        return self.child_win_value / (1 + self.child_number_visits)

    def child_U(self):
        return self.cpuct * np.sqrt(self.number_visits) * (
            self.child_priors / (1 + self.child_number_visits))

    def get_best_child(self):
        # This is slightly ugly, but it basically says that it should not, in a million years, visit the hypothetical "child-nodes"
        # that occur as a result of illegal moves.
        if self.player_number == 0:
            return np.unravel_index(np.argmin(self.child_Q() - self.child_U() + self.illegal_moves_mask * 100000), self.child_priors.shape)
        else:
            return np.unravel_index(np.argmax(self.child_Q() + self.child_U() - self.illegal_moves_mask * 100000), self.child_priors.shape)

    def get_move_to_make_for_search(self, ply_count):
        distribution = self.child_number_visits.flatten()
        # Play moves non-deterministically (weighted by distribution)
        # at first, then play the "best" afterwards
        if ply_count > self.nondet_plies:
            return np.unravel_index(np.argmax(distribution), self.child_priors.shape)
        else:
            distribution = softmax(distribution)
            return np.unravel_index(np.random.choice(np.arange(distribution.shape[0]), 1, p=distribution)[0], self.child_priors.shape)
