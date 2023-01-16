from functools import lru_cache

from montecarlo.node import Node

import numpy as np

from montecarlo.prediction_storage import Storage
from montecarlo.worker_message_handler import WorkerMessageHandler

class MonteCarlo:

    def __init__(self, thread: int, root_node: Node, all_moves, move_cap, dims, prior_noise_coefficient, cpuct, root_priors, root_value, net=None):
        self.thread = thread
        self.root_node = root_node
        self.move_cap = move_cap
        self.net = net
        self.all_moves = all_moves
        self.cnoise = prior_noise_coefficient
        self.cpuct = cpuct
        self.dims = dims
        self.storage = Storage()
        self.messager = WorkerMessageHandler(thread, root_priors.shape[0])
        self.root_priors = root_priors
        self.root_value = root_value

    def episode_done(self):
        self.storage.reset()
    
    def all_done(self):
        print("Waiting for all.")
        self.messager.wait_all()
        print("Done waiting for all.")


    def distribution(self):
        return self.root_node.child_number_visits

    def move_root(self, node):
        self.root_node = node
        self.root_node.parent = None

    def make_choice(self, ply_count):
        best_move = self.root_node.get_move_to_make_for_search(ply_count)
        if best_move not in self.root_node.children:
            self.root_node.add_child(best_move, self.root_node.child_priors[best_move])
        return self.root_node.children[best_move]

    def ask_for_neural_net_prediction(self, node):
        if node.eval_requested:
            return
        node.eval_requested = True
        label = self.storage.store(node)
        self.messager.send_state(node, label)

    def check_for_recieved_evals(self):
        results = self.messager.check_for_results()
        for result in results:
            label, (priors, value) = result
            node_to_update = self.storage.retrieve(label)
            if node_to_update is not None:
                self.revert_virtual_loss(node_to_update)
                self.apply_results_to_node(node_to_update, priors, value)
    
    def apply_results_to_node(self, node, child_priors, value_estimate):
        # The neural net is set to predict positively if the person whose turn it is has a good position.
        # 1 means winning for person to play, -1 means losing
        # The search, however, is set up so that -1 is good for black, and 1 is good for white, regardless of who's playing
        # So if the neural net predicts 1 and it's black's turn, this means that it thinks that black is winning, but the search
        # will agree if it is given -1.
        if node.state.turn == 0:
            value_estimate *= -1
        child_priors = np.reshape(child_priors, (self.dims[0], self.dims[1], self.move_cap))
        noise = np.random.uniform(size=self.dims[0] * self.dims[1] * self.move_cap)
        # Sum of noise is equal to 1
        noise = noise.reshape(child_priors.shape)

        noise = noise * (node.illegal_moves_mask == 0)

        noise /= noise.sum()

        child_priors = (1 - self.cnoise) * child_priors + self.cnoise * noise
        child_priors /= child_priors.sum()
        node.expand(child_priors)
        node.update_win_value(value_estimate)
    
    def revert_virtual_loss(self, node):
        loss = 1 if node.state.turn == 1 else -1
        node.update_win_value(loss)

    def apply_virtual_loss(self, node):
        loss = -1 if node.state.turn == 1 else 1
        node.update_win_value(loss)

    def simulate(self):
        # If it's just the root node, probably need to return an eval pretty quick...
        self.check_for_recieved_evals()

        leaf = self.root_node.select_leaf()
        # If the leaf is a terminal node, just return the actual result
        if leaf.state.game_result() is not None:
            leaf.update_win_value(leaf.state.game_result())
            return
        
        # If it is just the root node, return some pre-cached results
        # Can't really do anything while it's just the root node here
        if leaf is self.root_node:
            self.apply_results_to_node(leaf, self.root_priors, self.root_value)
            return
        
        self.apply_virtual_loss(leaf)
        self.ask_for_neural_net_prediction(leaf)

        
