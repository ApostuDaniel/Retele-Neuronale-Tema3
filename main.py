import gzip
import numpy
import pickle

import Utils
from NeuralNetwork import NeuralNetwork

with gzip.open(r'resources/mnist.pkl.gz', 'rb') as fd:
    train_set, valid_set, test_set = pickle.load(fd, encoding='latin')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    network = NeuralNetwork([784, 100, 10])
    epochs = network.training_phase(1, 32, train_set, valid_set, 0.05, Utils.sigmoid_activation, Utils.softmax_activation)
    best_epoch = max(epochs.values(), key=lambda x: x[1][0])
    print("Best epoch results on validation set: ", best_epoch[1])
    for idx, epoch in enumerate(epochs.values()):
        print("Epoch ", idx, ":", epoch[1])

