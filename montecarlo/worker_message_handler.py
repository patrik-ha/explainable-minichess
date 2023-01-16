import numpy as np

from .node import Node

from mpi4py import MPI

class WorkerMessageHandler:
    def __init__(self, thread, prior_size):
        self.thread = thread
        self.comm = MPI.COMM_WORLD
        self.GPU = 0
        # ([priors, value(s), label(s)])
        self.result_buffer_prototype = np.zeros((prior_size, 3), dtype=np.float32)
        self.active_requests = []

    
    def send_state(self, node: Node, label: int):
        state = node.state.copy()
        # Add a plane indicating what thread sent it
        # And what label it is given by that thread
        thread_plane = np.full_like(node.state[:, :, 0], self.thread)
        label_plane = np.full_like(node.state[:, :, 0], label)

        labeled_state = np.append(state, thread_plane, axis=-1)
        labeled_state = np.append(labeled_state, label_plane, axis=-1)

        req = self.comm.Isend([labeled_state, MPI.FLOAT], dest=self.GPU, tag=0)
        req.wait()
        
        buffer = np.zeros_like(self.result_buffer_prototype)
        recieve_request = self.comm.Irecv(buffer, self.GPU, tag=0)

        self.active_requests.append(recieve_request)

    def check_for_results(self):
        still_active = []
        results = []
        for req in self.active_requests:
            if req.Test():
                data = req.Wait()
                results.append(data)
            else:
                still_active.append(req)
        self.active_requests = still_active

        data_pairs = []
        for result in results:
            priors = result[0, :]
            value = results[1, 0]
            label = results[2, 0]
            data_pairs.append((label, (priors, value)))
        return data_pairs

