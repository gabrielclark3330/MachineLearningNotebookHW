import os, sys
import numpy as np
import math

from data import readDataLabels, normalize_data, train_test_split, to_categorical
from utils import accuracy_score

# Create an MLP with 8 neurons
# Input -> Hidden Layer -> Output Layer -> Output
# Neuron = f(w.x + b)
# Do forward and backward propagation

mode = 'train'      # train/test... Optional mode to avoid training incase you want to load saved model and test only.

class ANN:
    def __init__(self, num_input_features, num_hidden_units, num_outputs, hidden_unit_activation, output_activation, loss_function):
        self.num_input_features = num_input_features
        self.num_hidden_units = num_hidden_units
        self.num_outputs = num_outputs

        self.hidden_unit_activation = hidden_unit_activation
        self.output_activation = output_activation
        self.loss_function = loss_function


    def initialize_weights(self):   # TODO
        # Create and Initialize the weight matrices
        # Never initialize to all zeros. Not Cool!!!
        # Try something like uniform distribution. Do minimal research and use a cool initialization scheme.
        return

    def forward(self):      # TODO
        # x = input matrix
        # hidden activation y = f(z), where z = w.x + b
        # output = g(z'), where z' =  w'.y + b'
        # Trick here is not to think in terms of one neuron at a time
        # Rather think in terms of matrices where each 'element' represents a neuron
        # and a layer operation is carried out as a matrix operation corresponding to all neurons of the layer
        pass

    def backward(self):     # TODO
        pass

    def update_params(self):    # TODO
        # Take the optimization step.
        return

    def train(self, dataset, learning_rate=0.01, num_epochs=100):
        for epoch in range(num_epochs):
            pass

    def test(self, test_dataset):
        accuracy = 0    # Test accuracy
        # Get predictions from test dataset
        # Calculate the prediction accuracy, see utils.py
        return accuracy


def main(argv):
    ann = ANN()

    # Load dataset
    dataset = readDataLabels()      # dataset[0] = X, dataset[1] = y

    # Split data into train and test split. call function in data.py

    # call ann->train()... Once trained, try to store the model to avoid re-training everytime
    if mode == 'train':
        pass        # Call ann training code here
    else:
        # Call loading of trained model here, if using this mode (Not required, provided for convenience)
        raise NotImplementedError

    # Call ann->test().. to get accuracy in test set and print it.


if __name__ == "__main__":
    main(sys.argv)
