import tensorflow.keras as keras
import numpy as np
import chess
from tensorflow.keras import backend as K
from tqdm import tqdm

from sklearn.metrics import accuracy_score
import tensorflow as tf


class PredictorConvNet:
    def __init__(self, model):
        self.model = model

    def predict(self, boards):
        # with tf.device('/cpu:0'):
        # tensor = tf.convert_to_tensor(np.array(boards), dtype=tf.uint8)
        res = self.model.predict_single(boards)
        # tf.keras.backend.clear_session()
        policies, values = res
        return policies, values
