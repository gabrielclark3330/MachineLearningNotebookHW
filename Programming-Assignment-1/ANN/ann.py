import os, sys
import numpy as np
import math

from data import read_data_labels, normalize_data, train_test_split, to_categorical
from utils import accuracy_score, CrossEntropyLoss, SigmoidActivation, SoftmaxActivation, ReLUActivation, MSELoss

# Create an MLP with 8 neurons
# Input -> Hidden Layer -> Output Layer -> Output
# Neuron = f(w.x + b)
# Do forward and backward propagation

mode = 'train'  # train/test... Optional mode to avoid training incase you want to load saved model and test only.


class ANN:
    def __init__(self, num_input_features, num_hidden_units, num_outputs, hidden_unit_activation, output_activation,
                 loss_function):
        self.num_input_features = num_input_features
        self.num_hidden_units = num_hidden_units
        self.num_outputs = num_outputs

        self.hidden_weights = None
        self.output_weights = None
        self.hidden_bias = None
        self.output_bias = None

        self.hidden_unit_activation = hidden_unit_activation
        self.output_activation = output_activation
        self.loss_function = loss_function


    def initialize_weights(self):
        # The number of rows must equal the number of neurons in the previous layer. The number of columns must match the number of neurons in the next layer.
        # layers are 0 indexed with the input being layer 0
        # Input is a [1 X num_input_features](rows X cols) size matrix
        # inputw_eights doesn't exist because input doesn't have weights
        # hidden_weights is a [ X num_hidden_units] matrix
        # output_weights is a [num_outputs X 1] matrix
        # biases is an array [1 X num_layers-1] (3-1 in this case) because there is a single bias per layer

        # Example: 
        # input 5 | middle 10 | output 2
        # input [1 x 5] | middle [5 X 10] -> create [1 X 10] | output [10 X 2] -> create [1 X 2]
        # so input [1 X len(x)] | middle [len(x) X self.number_hidden_units] | output [self.number_hidden_units X self.number_outputs]

        self.hidden_bias = np.random.rand() - .5
        self.output_bias = np.random.rand() - .5
        self.hidden_weights = np.random.rand(self.num_input_features, self.num_hidden_units) - .5
        self.output_weights = np.random.rand(self.num_hidden_units, self.num_outputs) - .5


    def forward(self, x):  # TODO
        # x = input matrix
        # hidden activation y = f(z), where z = w.x + b
        # output = g(z'), where z' =  w'.y + b'
        # Trick here is not to think in terms of one neuron at a time
        # Rather think in terms of matrices where each 'element' represents a neuron
        # and a layer operation is carried out as a matrix operation corresponding to all neurons of the layer
        print("X", x)
        # where x is a [1 X self.num_input_features]
        zH = np.dot(x, self.hidden_weights) + self.hidden_bias
        print("zH", zH)
        neuron_activations = ReLUActivation().__call__(zH)
        print("relu activations", neuron_activations)

        # Output layer
        zO = np.dot(neuron_activations, self.output_weights) + self.output_bias
        print("zO", zO)
        neuron_activations = SoftmaxActivation().__call__(zO)
        print("softmax activations", neuron_activations)
        return neuron_activations


    def backward(self):  # TODO
        pass

    def update_params(self):  # TODO
        # Take the optimization step.
        return

    def train(self, dataset, learning_rate=0.01, num_epochs=100):
        self.initialize_weights()
        for epoch in range(num_epochs):
            for index in range(len(dataset[0])): # dataset[0] is the data dataset[1] is labels
                data = dataset[0][index]
                label = dataset[1][index]
                print("label",label)
                prediction = self.forward(data)
                vectorized_label = [1 if x==label else 0 for x in range(10)]
                print("Compare pred and label", prediction, vectorized_label)
                loss = MSELoss.__call__(self, prediction, vectorized_label)
                print(loss)
                break
            break

    def test(self, test_dataset):
        accuracy = 0  # Test accuracy
        # Get predictions from test dataset
        # Calculate the prediction accuracy, see utils.py
        accuracy_score(test_dataset[1], )  # TODO
        return accuracy


def main(argv):
    ann = ANN(64, 16, 10, SigmoidActivation(), SoftmaxActivation(), CrossEntropyLoss())

    # Load dataset
    dataset = read_data_labels()  # dataset[0] = X, dataset[1] = y
    dataset = normalize_data(dataset)

    # Split data into train and test split. call function in data.py
    #train[0] is the collection of training data and train[1] is the collection of training labels
    train, test = train_test_split(dataset[0], dataset[1])

    # call ann->train()... Once trained, try to store the model to avoid re-training everytime
    if mode == 'train':
        # Call ann training code here
        ann.train(train)
    else:
        # Call loading of trained model here, if using this mode (Not required, provided for convenience)
        raise NotImplementedError

    # Call ann->test().. to get accuracy in test set and print it.
    print('Accuracy:', ann.test(test))


if __name__ == "__main__":
    main(sys.argv)
