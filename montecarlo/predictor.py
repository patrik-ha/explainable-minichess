import numpy as np

from .node import Node

from mpi4py import MPI

class Predictor:
    def __init__(self, total_worker_threads, state_shape, net):
        self.total_worker_threads = total_worker_threads
        self.thread_offset = 2
        self.state_prototype = np.zeros((*state_shape[0:2], state_shape[2] + 2), dtype=np.float32)
        self.comm = MPI.COMM_WORLD
        self.initialize_receivers()
        self.initialize_halting_receiver()
        self.empty_storage() 
        
        self.net = net
        self.buffer_threshold = 256


    def work(self):
        received_states = []
        requests = []
        did_something = False
        for i, (req_buffer, req) in enumerate(self.requests):
            if req.Test():
                did_something = True
                req.Wait()
                received_states.append(req_buffer)
                new_buffer = np.zeros_like(self.state_prototype)
                new_request = self.comm.Irecv(new_buffer, i + self.thread_offset, tag=0)
                requests.append((new_buffer, new_request))
            else:
                requests.append((req_buffer, req))
        self.requests = requests
        
        for state in received_states:
            self.threads.append(state[0, 0, -2])
            self.labels.append(state[0, 0, -1])
            self.states.append(state[:, :, :-2])
        
        # Full buffer or didn't receive anything!
        if len(self.threads) > self.buffer_threshold or (not did_something and len(self.threads) > 0):
            # print(self.states)
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
        packet = np.zeros((3, priors.shape[0]), dtype=np.float32)
        packet[0] = priors
        packet[1] = value
        packet[2] = label
        return self.comm.Isend([packet, MPI.FLOAT], dest=thread, tag=0)
        
    
    def predict(self, states):
        priors, values = self.net.predict(np.array(states))
        return priors, values

    
    def initialize_receivers(self):
        self.requests = []
        for i in range(self.total_worker_threads):
            buffer = np.zeros_like(self.state_prototype)
            req = self.comm.Irecv(buffer, i + self.thread_offset, tag=0)
            self.requests.append((buffer, req))
        
    def initialize_halting_receiver(self):
        self.halting_buf = np.array([0])
        self.halting_req = self.comm.Irecv(self.halting_buf, source=0, tag=7)
    
    def should_stop(self):
        has_message = self.halting_req.Test()
        if has_message:
            self.halting_req.wait()
            for _, req in self.requests:
                MPI.Request.Cancel(req)
            return True
        return False