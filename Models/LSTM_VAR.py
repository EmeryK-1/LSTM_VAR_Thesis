import tensorflow as tf
from tensorflow.keras.layers import LSTM
from tensorflow.keras import Model
from tensorflow.keras.layers import RNN, LSTMCell
import numpy as np

def create_lstm_layer(units, return_sequences, name):
    cell = LSTMCell(units, name=f"{name}_cell")
    return RNN(cell, return_sequences=return_sequences, return_state=not return_sequences, name=name)


class LSTM_VAR(Model):
    def __init__(self, output_size, p, scaler, lstm_stack=1, units=64, linear=True, non_linear=True, backcast=True):
        super().__init__()
        self.output_size = output_size
        self.p = p
        self.units = units
        self.lstms = [create_lstm_layer(units=self.units, return_sequences=True, name=f'LSTM_{i}') for i in
                      range(lstm_stack - 1)]
        self.lstms.append(create_lstm_layer(units=self.units, return_sequences=False, name=f'LSTM_{lstm_stack}'))
        self.linear = linear
        self.non_linear = non_linear
        self.backcast = backcast
        self.scaler_scale = scaler.scale_
        self.scaler_min = scaler.min_

        self.F_l = self.add_weight(shape=(self.p * self.output_size, self.output_size), initializer='uniform',
                                   name='F_l')
        self.F_nl = self.add_weight(shape=(self.units, self.output_size), initializer='uniform', name='F_nl')
        self.F_nl_backcast = self.add_weight(shape=(self.units, self.p * self.output_size), initializer='uniform',
                                             name='F_nl_backcast')

    def build(self, input_shape):
        for lstm in self.lstms:
            input_shape = lstm.compute_output_shape(input_shape)

        self.built = True

    def call(self, input, training=True):
        # Create copy of input
        lstm_input = tf.identity(input)

        if self.scaler_scale is not None:
            # Reshape input to 2D
            reshaped_input = tf.reshape(input, [-1, input.shape[-1]])
            # Perform the scaling operation using TensorFlow
            scaled_input = reshaped_input * tf.constant(self.scaler_scale, dtype=tf.float32) + tf.constant(
                self.scaler_min, dtype=tf.float32)
            # Reshape the input back to 3D
            lstm_input = tf.reshape(scaled_input, tf.shape(input))

        for lstm in self.lstms[:-1]:  # all but the last one
            lstm_input = lstm(lstm_input, training=training)

        h, *state = self.lstms[-1](lstm_input, training=training)

        non_linear = tf.matmul(h, self.F_nl)
        # Flatten last dimension of input by putting all timesteps next to each other
        input = tf.reshape(input, (
            -1, input.shape[1] * input.shape[2]))  # First m inputs are lag 1, next m inputs are lag 2, etc.
        relevant_lags = input[:, :self.p * self.output_size]

        if self.backcast:
            input_backcast = tf.matmul(h, self.F_nl_backcast)
            relevant_lags = relevant_lags - input_backcast
        # Get last self.p*self.output_size inputs
        linear = tf.matmul(relevant_lags, self.F_l)
        if self.linear and self.non_linear:
            output = linear + non_linear
        elif self.linear:
            output = linear
        else:
            output = non_linear
        if not training:
            # reshape relevant_lags to 3D
            relevant_lags = tf.reshape(relevant_lags, (-1, self.p, self.output_size))
            return output, {'non_linear': non_linear, 'linear': linear, 'backcasted': relevant_lags}
        return output

    def get_var_matrix(self):
        return np.array(self.F_l).T