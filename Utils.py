import math

import numpy
import numpy as np

import numpy as np


def mini_batch_training(training_set, batch_size, weights, biases, learning_rate):
    batch_start = 0
    while True:
        nr_of_rows = min(batch_size, len(training_set[0]) - batch_start)
        mini_batch = training_set[0][batch_start:batch_start + nr_of_rows]
        mini_batch_actual_values = training_set[1][batch_start:batch_start + nr_of_rows]
        batch_start += nr_of_rows
        t = np.array([number_to_number_array(x, len(biases[0])) for x in mini_batch_actual_values])
        z = (mini_batch @ weights) + biases
        y = activation(z)
        diff = t - y
        delta = (mini_batch.T @ diff) * learning_rate / len(training_set[0])
        beta = np.sum(diff * learning_rate, axis=0) * learning_rate / len(
            training_set[0])  # pentru a aduce de la 32,10 la 1,10
        weights += delta
        biases += beta
        if batch_start >= len(training_set[0]):
            break
    # np.random.shuffle(training_set[0])
    shuffle_in_unison(training_set[0], training_set[1])


def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

    # Creates an array of length array_len filled with 0, but the value at index is 1


def number_to_number_array(index, array_len):
    digit_array = [0] * array_len
    digit_array[index] = 1
    return digit_array


def sigmoid_activation(z: float):
    return 1 / 1 + math.e ** (-z)


def softmax_activation(neuron_idx: int, neuron_outputs: numpy.ndarray):
    return (math.e ** neuron_outputs[neuron_idx]) / sum([math.e ** z for z in neuron_outputs])


def activation(result_matrix):
    activation_matrix = []
    for row in result_matrix:
        activation_row = [0 if x <= 0 else 1 for x in row]
        activation_matrix.append(activation_row)
    return np.array(activation_matrix)
