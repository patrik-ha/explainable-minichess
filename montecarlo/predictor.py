import numpy as np

from .node import Node

from mpi4py import MPI

class Predictor:
    def __init__(self, total_worker_threads, state_shape, net):
        self.total_worker_threads = total_worker_threads
        self.thread_offset = 1
        self.state_prototype = np.zeros((*state_shape[0:2], state_shape[2] + 2), dtype=np.float32)
        self.comm = MPI.COMM_WORLD
        self.initialize_receivers()
        self.empty_storage() 
        
        self.net = net

        self.buffer_threshold = 32


    def work(self):
        received_states = []
        requests = []
        for i, req in self.requests:
            if req.Test():
                received_states.append(req.Wait())
                # Make a new preemptive receive-request
                buffer = np.zeros_like(self.state_prototype)
                new_request = self.comm.Irecv(buffer, i + self.thread_offset, tag=0)
                requests.append(new_request)
            else:
                requests.append(req)
        self.requests = requests
        
        for state in received_states:
            self.threads.append(state[:, :, -2])
            self.labels.append(state[:, :, -1])
            self.states.append(state[:, :, :-2])
        
        if len(self.threads) > self.buffer_threshold:
            self.perform_batch_prediction()
            self.empty_storage()

    
    def empty_storage(self):
        self.threads = []
        self.labels = []
        self.states = []
    
    def perform_batch_prediction(self):
        priors, values = self.predict(self.states)
        send_reqs = []
        for i, label, thread in zip(range(len(self.threads)), self.labels, self.threads):
            send_reqs.append(self.send_results(thread, label, priors[i], values[i]))
        MPI.Request.Waitall(send_reqs)


    def send_results(self, thread, label, priors, value):
        packet = np.zeros((priors.shape[0], 3), dtype=np.float32)
        packet[0] = priors
        packet[1] = value
        packet[2] = label
        return self.comm.Isend([packet, MPI.FLOAT], dest=thread, tag=0)
        
    
    def predict(self, states):
        priors, values = self.net(states)
        return priors, values

    
    def initialize_receivers(self):
        self.requests = []
        for i in range(self.total_worker_threads):
            buffer = np.zeros_like(self.state_prototype)
            req = self.comm.Irecv(buffer, i + self.thread_offset, tag=0)
            self.requests.append(req)
        
