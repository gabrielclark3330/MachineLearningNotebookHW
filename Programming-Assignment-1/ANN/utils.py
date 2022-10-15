import numpy as np
import math


class MSELoss:      # For Reference
    def __init__(self):
        # Buffers to store intermediate results.
        self.current_prediction = None
        self.current_gt = None
        pass

    def __call__(self, y_pred, y_gt):
        self.current_prediction = y_pred
        self.current_gt = y_gt

        # MSE = 0.5 x (GT - Prediction)^2
        loss = 0.5 * np.power((y_gt - y_pred), 2)
        return loss

    def grad(self):
        # Derived by calculating dL/dy_pred
        gradient = -1 * (self.current_gt - self.current_prediction)

        # We are creating and emptying buffers to emulate computation graphs in
        # Modern ML frameworks such as Tensorflow and Pytorch. It is not required.
        self.current_prediction = None
        self.current_gt = None

        return gradient


class CrossEntropyLoss:     # TODO: Make this work!!!
    def __init__(self):
        # Buffers to store intermediate results.
        self.current_prediction = None
        self.current_gt = None
        pass

    def __call__(self, y_pred, y_gt):
        # TODO: Calculate Loss Function
        loss = None
        return loss

    def grad(self):
        # TODO: Calculate Gradients for back propagation
        gradient = None
        return gradient


class SoftmaxActivation:   
    def __init__(self):
        self.arr = [] # added to be used in the call and grad
        pass

    def __call__(self, y):
        e = exp(vector)
        return e / e.sum()
        '''
        # assumed that y is a np array. may need to fix later.
        exponeitate = np.exp(y)
        probs = exponeitate / np.sum(exponeitate)
        summ = probs.sum() # the sum
        self.arr = summ
        return summ
        '''

    def __grad__(self):
        # assumed that we are using the calcuated summ from the call function. 
        arr = self.arr
        jacobianMatrix = np.diag(arr)
        length = len(jacobianMatrix)
        for i in range(length):
            for j in range(length):
                if i == j:
                    jacobianMatrix[i][j] = arr[i] * (1-arr[i])
                else:
                    jacobianMatrix[i][j] = -arr[i] * arr[j]
        
        return jacobianMatrix 


class SigmoidActivation:    # TODO: Make this work!!!
    def __init__(self):
        pass

    def __call__(self, y):
        # TODO: Calculate Activation Function
        pass

    def __grad__(self):
        # TODO: Calculate Gradients.. Remember this is calculated w.r.t. input to the function -> dy/dz
        pass


class ReLUActivation:
    def __init__(self):
        self.z = None
        pass

    def __call__(self, z):
        # y = f(z) = max(z, 0) -> Refer to the computational model of an Artificial Neuron
        self.z = z
        y = np.maximum(z, 0)
        return y

    def __grad__(self):
        # dy/dz = 1 if z was > 0 or dy/dz = 0 if z was <= 0
        gradient = np.where(self.z > 0, 1, 0)
        return gradient


def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy
