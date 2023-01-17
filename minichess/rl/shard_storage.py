import numpy as np

class ShardStorage:
    def __init__(self, buffer_size=1000000):
        self.buffer_size = buffer_size
        self.shards_received = 0
        self.reset_storage()
    
    def reset_storage(self):
        self.state_buffer = []
        self.distribution_buffer = []
        self.value_buffer = []
        self.outcomes = []

    def extend_storage(self, states, distributions, values, outcomes):
        self.state_buffer.extend(states)
        for dist in distributions:
            self.distribution_buffer.append(dist / np.sum(dist))
        self.value_buffer.extend(values)
        self.outcomes.extend(outcomes)
        self.shards_received += 1

    def truncate_storage(self):
        self.state_buffer = self.state_buffer[-self.buffer_size:]
        self.distribution_buffer = self.distribution_buffer[-self.buffer_size:]
        self.value_buffer = self.value_buffer[-self.buffer_size:]
        self.outcomes = self.outcomes[-self.buffer_size:]
    
    def get_training_samples(self):
        return np.array(self.state_buffer), np.array(self.distribution_buffer), np.array(self.value_buffer)