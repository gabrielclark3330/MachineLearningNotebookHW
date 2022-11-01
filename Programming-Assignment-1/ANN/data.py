import os, sys
import numpy as np
from sklearn import datasets

#import libraries as needed

def sigmoid(x):
	# np.exp(-x) = e^-x
	y = 1 / (1 + np.exp(-x))
	return y

def softmax(x):
	# Need to add 1 * 10^-15 to fix numerical instability (where we get 0)
	y = np.exp(x) / np.sum(np.exp(x), axis=1)
	return y

def readDataLabels(): 
	#read in the data and the labels to feed into the ANN
	data = datasets.load_digits()
	X = data.data
	y = data.target

	# This creates our expected output with a row with 10 where all are 0 but the one that is correct
	y_gt = np.zeros((y.shape[0], np.amax(y) + 1))

	# Normalization
	X_norm = (X - np.amin(X)) / (np.amax(X) - np.amin(X))

	# Split training and test
	ratio = 0.6
	num_train_samples = int(y.shape[0] * ratio)
	x_train, y_train = X_norm[:num_train_samples], y_gt[:num_train_samples]
	x_test, y_test = X_norm[num_train_samples:], y_gt[num_train_samples:]
	print(y_train.shape, y_test.shape)  # (1078, 10) (719, 10)
	# The issue here is that the data is not shuffled. This could mean if the data
	# that we are getting might have a bias from the person that put the data together
	# or even by accident which will affect the bias.

	# Initialize Params
	w1 = np.ones((64, 15))
	b1 = np.zeros((1, 16))

	w2 = np.ones((16, 10))
	b2 = np.zeros((1, 10))

	# Forward
	z1 = x_train.dot(w1) + b1
	print(z1.shape)  # (1078, 16)
	a1 = sigmoid(z1)
	print(a1.shape)  # (1078, 16)

	return X,y

def to_categorical(y):
	
	#Convert the nominal y values tocategorical

	return y
	
def train_test_split(data,labels,n=0.8): #TODO

	#split data in training and testing sets

	return 

def normalize_data(data): #TODO
	# normalize/standardize the data
	# We do this because our variance should be from 0 to 1. Right now it is from 0 to 16.
	# The neurons saturate from 0 to 1 so anything greater will over saturate.
	return (data - np.amin(data)) / (np.amax(data) - np.amin(data))
