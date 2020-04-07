from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow import keras
from string import Template
import tensorflow as tf
import numpy as np


def copy_dense(model: Dense):

    dense = Dense(units=model.units, input_shape=model.input_shape,
                  activation=model.activation, kernel_initializer=model.kernel_initializer)
    dense.build(input_shape=model.input_shape)
    dense.kernel = tf.identity(model.kernel)
    dense.bias = tf.identity(model.bias)
    dense._trainable_weights.append(dense.kernel)
    dense._trainable_weights.append(dense.bias)
    return dense


def copy_conv(model):
    pass


def copy_dropout(model):
    pass


def copy_a_sequential(seq: Sequential):
    new_seq = Sequential()
    for l in seq.layers:
        if type(l) == Dense:
            new_seq.add(copy_dense(l))

    return new_seq

