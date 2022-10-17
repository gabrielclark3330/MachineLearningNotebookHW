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

        self.hidden_bias = np.random.rand(self.num_hidden_units, 1) - 0.5
        self.output_bias = np.random.rand(self.num_outputs, 1) - 0.5
        self.hidden_weights = np.random.rand(self.num_hidden_units, self.num_input_features) - 0.5
        self.output_weights = np.random.rand(self.num_outputs, self.num_hidden_units) - 0.5


    def forward(self, x):
        # x = input matrix or entire dataset
        # hidden activation y = f(z), where z = w.x + b
        # output = g(z'), where z' =  w'.y + b'
        # Trick here is not to think in terms of one neuron at a time
        # Rather think in terms of matrices where each 'element' represents a neuron
        # and a layer operation is carried out as a matrix operation corresponding to all neurons of the layer
        # where x is a [1 X self.num_input_features]
        zH = self.hidden_weights.dot(x) + self.hidden_bias
        aH = self.hidden_unit_activation.__call__(zH)

        # Output layer
        zO = self.output_weights.dot(aH) + self.output_bias
        aO = self.output_activation.__call__(zO)
        return zH, aH, zO, aO

    def backward(self, number_data_samples, zH, aH, zO, aO, x, y): # TODO
        wH = self.hidden_weights
        wO = self.output_weights
        dZO = aO - np.array(y).T
        #dZO = self.loss_function.__call__(aO, y)
        dWO = 1 / number_data_samples * dZO.dot(aH.T)
        dBO = 1 / number_data_samples * np.sum(dZO)
        dZH = wO.T.dot(dZO) * self.hidden_unit_activation.__grad__() #zH
        dWH = 1 / number_data_samples * dZH.dot(x.T)
        dBH = 1 / number_data_samples * np.sum(dZH)
        return dWH, dBH, dWO, dBO

    def update_params(self, wH, bH, wO, bO, dWH, dBH, dWO, dBO, alpha):
        # Take the optimization step.
        temp_wH = wH - alpha * dWH
        temp_bH = bH - alpha * dBH
        temp_wO = wO - alpha * dWO
        temp_bO = bO - alpha * dBO
        return temp_wH, temp_bH, temp_wO, temp_bO

    def train(self, dataset, learning_rate=0.1, num_epochs=100): # TODO
        self.initialize_weights()
        for epoch in range(num_epochs):
            zH, aH, zO, aO = self.forward(dataset[0])
            onehot_labels = [[1 if x==label else 0 for x in range(10)] for label in dataset[1]]
            #print("Compare pred and label", aO, onehot_labels)
            #loss = MSELoss.__call__(self, aO, onehot_labels)
            #print("loss", loss)
            number_data_samples = len(dataset[0])
            dWH, dBH, dWO, dBO = self.backward(number_data_samples, zH, aH, zO, aO, dataset[0], onehot_labels)
            #print(wH, bH, wO, bO)
            self.hidden_weights, self.hidden_bias, self.output_weights, self.output_bias = \
                self.update_params(self.hidden_weights, self.hidden_bias, self.output_weights, self.output_bias, dWH, dBH, dWO, dBO, learning_rate)
            print(self.test(dataset))

    def test(self, test_dataset):
        accuracy = 0  # Test accuracy
        # Get predictions from test dataset
        # Calculate the prediction accuracy, see utils.py
        zH, aH, zO, aO = self.forward(test_dataset[0])
        #print(sum(aO[0]))
        guesses = [list(x).index(max(x)) for x in aO.T]
        accuracy = accuracy_score(test_dataset[1], guesses)
        return accuracy


def main(argv):
    ann = ANN(64, 16, 10, SigmoidActivation(), SoftmaxActivation(), MSELoss())

    # Load dataset
    dataset = read_data_labels()  # dataset[0] = X, dataset[1] = y

    # Split data into train and test split. call function in data.py
    #train[0] is the collection of training data and train[1] is the collection of training labels
    train, test = train_test_split(dataset[0], dataset[1])

    norm_train = [[], train[1]]
    norm_train[0] = normalize_data(train[0])
    norm_train[0] = norm_train[0].T
    norm_test = [[], test[1]]
    norm_test[0] = normalize_data(test[0])
    norm_test[0] = norm_test[0].T

    # call ann->train()... Once trained, try to store the model to avoid re-training everytime
    if mode == 'train':
        # Call ann training code here
        ann.train(norm_train)
    else:
        # Call loading of trained model here, if using this mode (Not required, provided for convenience)
        raise NotImplementedError

    # Call ann->test().. to get accuracy in test set and print it.
    print('Accuracy:', ann.test(norm_test))


if __name__ == "__main__":
    main(sys.argv)
