import tensorflow.keras as keras
import numpy as np
import chess
from tensorflow.keras import backend as K
from tqdm import tqdm

from sklearn.metrics import accuracy_score
import tensorflow as tf


class ConvNet:
    def __init__(self, input_shape, move_cap, init=True):
        BLOCK_FILTER_SIZE = 32
        if not init:
            return
        position_input = keras.Input((input_shape))
        # Essentially make a list of matrices, which is an equivalent representation
        base = keras.layers.Conv2D(BLOCK_FILTER_SIZE, (3, 3), activation="elu", padding="same", name="res_block_output_base")(position_input)
        block_amount = 1
        base = keras.layers.Conv2D(BLOCK_FILTER_SIZE, (3, 3), activation="elu", padding="same")(base)
        base = keras.layers.Conv2D(BLOCK_FILTER_SIZE, (3, 3), activation="elu", padding="same")(base)

        # Policy head
        policy = keras.layers.Conv2D(move_cap, (1, 1), activation="elu", padding="same")(base)
        policy = keras.layers.Flatten()(policy)
        policy_output = keras.layers.Softmax(name="policy_output")(policy)

        # Value head

        val = keras.layers.Conv2D(16, (1, 1), name="value_conv", activation="elu", padding="same")(base)
        val = keras.layers.Flatten()(val)
        # val = keras.layers.Dense(256, activation="elu")(val)
        value_output = keras.layers.Dense(1, name="value_output", activation="tanh")(val)

        self.model = keras.Model(position_input, [policy_output, value_output])
        self.model.summary()
        self.model.compile(
            loss={"policy_output": keras.losses.CategoricalCrossentropy(), "value_output": keras.losses.MeanSquaredError()},
            loss_weights={"policy_output": 1.0, "value_output": 1.0},
            optimizer=keras.optimizers.Adam(learning_rate=0.001))

    def get_all_resblock_outputs(self, boards):
        """Returns a model that gives the activations from resnet-blocks"""
        if len(boards.shape) == 3:
            boards = np.reshape(boards, (1, *boards.shape))

        # All inputs
        inp = self.model.input
        # All outputs of the residual blocks
        outputs = [layer.output for layer in self.model.layers if "conv" in layer.name]
        functor = K.function([inp], outputs)

        BATCH_SIZE = 32
        all_layer_outs = []
        for i in tqdm(range(0, boards.shape[0], BATCH_SIZE)):
            layer_outs = functor([boards[i:i + BATCH_SIZE]])
            all_layer_outs.append(layer_outs)

        return all_layer_outs

    def fit(self, states, distributions, values, epochs=10):
        with tf.device('/gpu:0'):
            return self.model.fit(states, [distributions, values], epochs=epochs, batch_size=128)

    def predict(self, boards):
        if len(boards.shape) == 3:
            boards = np.reshape(boards, (1, *boards.shape))
        with tf.device('/cpu:0'):
            res = self.model(boards, training=False)
        policies, values = res
        return policies, values

    def predict_multi(self, boards):
        # with tf.device('/cpu:0'):
        # tensor = tf.convert_to_tensor(np.array(boards), dtype=tf.uint8)
        res = self.model.predict(np.array(boards))
        # tf.keras.backend.clear_session()
        policies, values = res
        return policies, values
