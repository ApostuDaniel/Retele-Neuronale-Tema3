import numpy as np
from typing import Callable

import Utils
from Layer import Layer
from Utils import number_to_number_array


class NeuralNetwork:
    layers: list[Layer] = []
    layer_sizes: list[int]

    def __init__(self, layer_sizes: list[int]):
        self.layer_sizes = layer_sizes
        for idx, input_size in enumerate(layer_sizes[:-1]):
            # the current layers nr of outputs(nodes) is the next layers number of inputs
            self.layers += [Layer(input_size, layer_sizes[idx + 1])]

    def make_copy(self):
        copy = NeuralNetwork(self.layer_sizes)
        for idx, layer in enumerate(self.layers):
            layer_in_copy = copy.layers[idx]
            layer_in_copy.weights = np.copy(layer.weights)
            layer_in_copy.biases = np.copy(layer.biases)
        return copy

    def calculate_outputs(self, training_set: np.ndarray, activation_function: Callable[[float], float],
                          output_activation_function: Callable):
        training_input = training_set
        for layer in self.layers[:-1]:
            output = layer.calculate_output(training_input)
            layer.activation_for_layer(output, activation_function)
            training_input = layer.layer_activation
        output_layer = self.layers[-1]
        output = output_layer.calculate_output(training_input)
        output_layer.activation_for_output(output, output_activation_function)

        return output_layer.layer_activation

    def calculate_last_layer_error(self, output_activation: np.ndarray, expected_labels: np.ndarray):
        t = np.array([number_to_number_array(x, self.layers[-1].nr_of_neurons) for x in expected_labels])
        last_layer_error = output_activation - t
        self.layers[-1].layer_error = last_layer_error
        return last_layer_error

    def backpropagation(self, training_set: np.ndarray, expected_labels: np.ndarray, learning_rate: float,
                        activation_function: Callable[[float], float],
                        output_activation_function: Callable):
        output_layer_activation = self.calculate_outputs(training_set, activation_function, output_activation_function)
        last_layer_error = self.calculate_last_layer_error(output_layer_activation, expected_labels)

        for layer_idx in reversed(range(len(self.layers) - 1)):
            layer = self.layers[layer_idx]
            subsequent_layer = self.layers[layer_idx + 1]
            layer_error = layer.layer_activation * (1 - layer.layer_activation) * \
                          (subsequent_layer.layer_error @ subsequent_layer.weights.T)
            layer.layer_error = layer_error
            subsequent_layer.cost_gradient_w = subsequent_layer.layer_error * layer.layer_activation
            subsequent_layer.cost_gradient_b = np.sum(subsequent_layer.layer_error, axis=0)
            subsequent_layer.weights += learning_rate * subsequent_layer.cost_gradient_w
            subsequent_layer.biases += learning_rate * subsequent_layer.cost_gradient_b

    def training_phase(self, nr_of_iterations: int, batch_size: int, training_data: np.ndarray,
                       validation_data: np.ndarray,
                       learning_rate: float,
                       activation_function: Callable[[float], float],
                       output_activation_function: Callable):
        epochs = {}
        iteration = 1
        while iteration <= nr_of_iterations:
            batch_start = 0
            while True:
                nr_of_rows = min(batch_size, len(training_data[0]) - batch_start)
                training_set = training_data[0][batch_start:batch_start + nr_of_rows]
                expected_labels = training_data[1][batch_start:batch_start + nr_of_rows]
                batch_start += nr_of_rows
                self.backpropagation(training_set, expected_labels, learning_rate, activation_function,
                                     output_activation_function)
                if batch_start >= len(training_set[0]):
                    break
            current_epoch_network = self.make_copy()
            epochs[iteration] = (current_epoch_network, current_epoch_network.test_network(validation_data))
            iteration += 1
        return epochs

    def test_network(self, test_data: np.ndarray):
        accurate = 0
        inaccurate = 0
        for index in range(len(test_data[0])):
            coordinates = test_data[0][index]
            actual_value = test_data[1][index]
            prediction_probabilities = self.calculate_outputs(coordinates, Utils.sigmoid_activation,
                                                              Utils.softmax_activation)
            prediction = np.argmax(prediction_probabilities)
            if actual_value == prediction:
                accurate += 1
            else:
                inaccurate += 1
        return accurate, inaccurate
