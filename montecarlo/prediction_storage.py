import numpy as np

from .node import Node

class Storage:
    def __init__(self):
        self.MAX_LABEL = 10000
        self.ind = 0
        self.nodes = [None] * self.MAX_LABEL

    
    def store(self, node: Node):
        label = self.ind
        self.nodes[label] = node
        self.ind += 1
        self.ind %= self.MAX_LABEL
        return label
    
    def retrieve(self, label):
        # The search thread has recieved some label for some state
        # Need to know what node it corresponds to...
        return self.nodes[label]

