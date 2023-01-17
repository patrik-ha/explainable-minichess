import os
import tensorflow.summary as summary
import string
import random
import numpy as np

def random_string(n):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=n))

class TensorboardWriter:
    def __init__(self, use_tensorboard, full_name, model_name):
        self.use_tensorboard = use_tensorboard
        self.full_name = full_name
        self.model_name = model_name
        if not use_tensorboard:
            return
        self.create_directory_for_run()
        self.launch_tensorboard()
    
    def create_directory_for_run(self):
        self.base_summary_dir = os.path.join(os.getcwd(), "tensorboard_logs", self.full_name, self.model_name)
        self.full_summary_dir = os.path.join(self.base_summary_dir, random_string(5))
        self.summary_writer = summary.create_file_writer(self.full_summary_dir)
        
    
    def launch_tensorboard(self):
        from tensorboard import program
        tb = program.TensorBoard()
        tb.configure(argv=[None, "--logdir", self.base_summary_dir])
        url = tb.launch()
        print(f"Tensorboard listening on {url}")
    
    def write_outcomes(self, outcomes, epoch):
        if not self.use_tensorboard:
            return

        outcome_array = np.array(outcomes)
        with self.summary_writer.as_default():
            summary.text("Simulation game results", "W-D-B: {}-{}-{}".format((outcome_array == 1).sum(), (outcome_array == 2).sum(), (outcome_array == 0).sum()), step=epoch)
    

    def write_losses(self, history, epoch):
        if not self.use_tensorboard:
            return

        with self.summary_writer.as_default():
            for loss in ["loss", "value_output_loss", "policy_output_loss"]:
                summary.scalar(name=loss, data=history.history[loss][0], step=epoch)
        self.summary_writer.flush()
    
    def write_checkpoint_message(self, checkpoint_path, epoch):
        if not self.use_tensorboard:
            return

        with self.summary_writer.as_default():
            summary.text("Checkpoints", "Checkpoint made at {}.".format(checkpoint_path), step=epoch)
        self.summary_writer.flush()