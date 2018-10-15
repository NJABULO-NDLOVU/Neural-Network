import numpy as np
import scipy.special
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image

class neuralNetwork:

	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
		#set the number of nodes
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes

		#learning rate
		self.lr = learningrate

		#link weights; where i = input, h = hidden, o = output
		self.wih = (np.random.rand(self.hnodes,self.inodes)-0.5)
		self.who = (np.random.rand(self.onodes,self.hnodes)-0.5)

		#activation function of sigmoid function
		self.activation_function = lambda x: scipy.special.expit(x)



	def train(self, inputs_list, targets_list):
		# convert inputs list to 2d array
		inputs = np.array(inputs_list, ndmin=2).T
		targets = np.array(targets_list, ndmin=2).T
		# calculate signals into hidden layer
		hidden_inputs = np.dot(self.wih, inputs)
		# calculate the signals emerging from hidden layer
		hidden_outputs = self.activation_function(hidden_inputs)
		# calculate signals into final output layer
		final_inputs = np.dot(self.who, hidden_outputs)
		# calculate the signals emerging from final output layer
		final_outputs = self.activation_function(final_inputs)

		#error 
		output_errors = targets - final_outputs

		# hidden layer error is the output_errors, split by weights, recombined at hidden nodes
		hidden_errors = np.dot(self.who.T, output_errors)

		# update the weights for the links between the hidden and output layers
		self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
		np.transpose(hidden_outputs))

		# update the weights for the links between the hidden and output layers
		self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
		np.transpose(hidden_outputs))
		# update the weights for the links between the input and hidden layers
		self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
		np.transpose(inputs))

	def query(self, inputs_list):
		#converts inputs list to 2d array
		inputs = np.array(inputs_list, ndmin = 2).T

		#calculate signals into hidden layer
		hidden_inputs = np.dot(self.wih, inputs)
		#calculate the signals emerging from hidden layer
		hidden_outputs = self.activation_function(hidden_inputs)

		#calculate signals into final output layer
		final_inputs = np.dot(self.who, hidden_outputs)
		#calculate the signals emrging from final output
		final_outputs = self.activation_function(final_inputs)

		return final_outputs




