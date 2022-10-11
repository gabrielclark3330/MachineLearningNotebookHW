import os, sys
import numpy as np
from sklearn import datasets

#import libraries as needed

def readDataLabels(): 
	#read in the data and the labels to feed into the ANN
	data = datasets.load_digits()
	X = data.data
	y = data.target

	return X,y

def to_categorical(y):
	
	#Convert the nominal y values tocategorical

	return y
	
def train_test_split(data,labels,n=0.8): #TODO

	#split data in training and testing sets

	return 

def normalize_data(data): #TODO

	# normalize/standardize the data

	return
