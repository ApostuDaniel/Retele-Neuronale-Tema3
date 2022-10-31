import math

import numpy
import numpy as np
from typing import Callable


class Layer:
    nr_of_inputs: int
    nr_of_neurons: int
    cost_gradient_w: np.ndarray
    cost_gradient_b: np.ndarray
    weights: np.ndarray
    biases: np.ndarray
    layer_activation: np.ndarray
    layer_error: np.ndarray

    def __init__(self, nr_nodes_in: int, nr_nodes_out: int):
        self.nr_of_inputs = nr_nodes_in
        self.nr_of_neurons = nr_nodes_out
        self.weights = np.random.normal(0, 1 / math.sqrt(nr_nodes_in), size=(self.nr_of_inputs, self.nr_of_neurons))
        self.biases = np.random.rand(1, self.nr_of_neurons)
        self.cost_gradient_w = np.zeros((self.nr_of_inputs, self.nr_of_neurons))
        self.cost_gradient_b = np.zeros((1, self.nr_of_neurons))

    def calculate_output(self, training_input: np.ndarray):
        z = (training_input @ self.weights) + self.biases
        return z

    def activation_for_layer(self, output: np.ndarray, activation_function: Callable[[float], float]):
        activation = []
        for row in output:
            activation += [[activation_function(z) for z in row]]
        self.layer_activation = numpy.array(activation)

    def activation_for_output(self, output: np.ndarray, output_activation_function: Callable):
        activation = []
        for row in output:
            activation += [[output_activation_function(idx, row) for idx, z in enumerate(row)]]
        self.layer_activation = numpy.array(activation)
