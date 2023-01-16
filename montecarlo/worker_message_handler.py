import numpy as np

from .node import Node

from mpi4py import MPI

class WorkerMessageHandler:
    def __init__(self, thread, prior_size):
        self.thread = thread
        self.comm = MPI.COMM_WORLD
        self.GPU = 1
        # ([priors, value(s), label(s)])
        self.result_buffer_prototype = np.zeros((3, prior_size), dtype=np.float32)
        self.active_requests = []


    def wait_all(self):
        MPI.Request.Waitall([r for (_, r) in self.active_requests])

    
    def send_state(self, node: Node, label: int):
        state = node.state.agent_board_state().copy()
        # Add a plane indicating what thread sent it
        # And what label it is given by that thread
        thread_plane = np.expand_dims(np.full_like(state[:, :, 0], self.thread), axis=-1)
        label_plane = np.expand_dims(np.full_like(state[:, :, 0], label), axis=-1)

        labeled_state = np.append(state, thread_plane, axis=-1)
        labeled_state = np.append(labeled_state, label_plane, axis=-1)

        req = self.comm.Isend([labeled_state, MPI.FLOAT], dest=self.GPU, tag=0)
        req.wait()
        # print("Just sent")
        # print(labeled_state)
        
        buffer = np.zeros_like(self.result_buffer_prototype)
        receive_request = self.comm.Irecv(buffer, self.GPU, tag=0)

        self.active_requests.append((buffer, receive_request))

    def check_for_results(self):
        still_active = []
        results = []
        for buffer, req in self.active_requests:
            if req.Test():
                req.Wait()
                results.append(buffer)
            else:
                still_active.append((buffer, req))
        self.active_requests = still_active

        data_pairs = []
        for result in results:
            priors = result[0, :]
            value = result[1, 0]
            label = int(result[2, 0])
            data_pairs.append((label, (priors, value)))
        return data_pairs
