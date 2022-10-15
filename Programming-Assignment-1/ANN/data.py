import math
import os, sys
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets


# import libraries as needed

def read_data_labels():
	# read in the data and the labels to feed into the ANN
	data = datasets.load_digits()

	'''
	# Visualize the data
	print('Shape:', data.data.shape)
	plt.gray()
	plt.matshow(data.images[0])
	plt.show()
	'''

	x = data.data
	y = data.target

	return x, y


def to_categorical(y):
	# Convert the nominal y values to categorical

	return y


def train_test_split(data, labels, n=0.8):
	assert len(data) == len(labels), 'Data and labels are not the same length!'

	# split data in training and testing sets
	split = math.floor(len(data) * n)
	return [data[:split], labels[:split]], [data[split:], labels[split:]]


def normalize_data(data):  
	# normalize/standardize the data
	# assuming np array.
	# assumption: used min max normalization. And normalized x and y seperately.
	x = data[0]
	y = data[1]
	normalizedData = [[],[]]

	xNormalized = (x - x.min()) / (x.max() - x.min())
	yNormalized = (y - y.min()) / (y.max() - y.min())

	normalizedData[0] = xNormalized
	normalizedData[1] = yNormalized

	return normalizedData
