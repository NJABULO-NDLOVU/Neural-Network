import neural as nu
from PIL import Image
import numpy as np
import scipy.misc
import scipy.special
import matplotlib.pyplot as plt
import dill as pickle

"""
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
# learning rate is 0.3
learning_rate = 0.3
# create instance of neural network


#TRAINING
data_file = open('C:\\Users\\NJABULO\\Downloads\\Data\\mnist_train.csv', 'r')
data_list = data_file.readlines()
data_file.close()


n = nu.neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

for record in data_list:
	all_values = record.split(';')
	inputs = (np.asfarray(all_values[1:])/255.0*0.99)+0.01
	targets = np.zeros(output_nodes)+0.01
	targets[int(all_values[0])] = 0.99
	n.train(inputs, targets)
pickle_out = open('Mnist.pickle', 'wb')
pickle.dump(n,pickle_out)
pickle_out.close()
"""
pickle_in = open('Mnist.pickle', 'rb')
p = pickle.load(pickle_in)
#TEst
"""
test_data_file = open('C:\\Users\\NJABULO\\Downloads\\Data\\mnist_test.csv', 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
np.savez_compressed('mnist_list_test', test_data_list)
test_data_list = np.load('mnist_list_test.npz')
"""

test_data_list = test_data_list['arr_0']
"""
im_array = scipy.misc.imread('C:\\Users\\NJABULO\\Downloads\\Data\\zero.png', flatten=True)
im_data = 255.0 -im_array.reshape(784)
im = np.asfarray(im_data).reshape((28,28))
im_data = (im_data/255.0*0.99)+0.01
"""
count = 0
for i in range(0,1000):	
	all_values = test_data_list[i].split(',')
	image_array = np.asfarray(all_values[1:]).reshape((28,28))
	p_guess = p.query((np.asfarray(all_values[1:])/255.0*0.99)+0.01)
	if int(p_guess.argmax()) == int(all_values[0]):
		count = count + 1
print((count/1000)*100)
#94.5
#The neural network has 94.5% accuracy with only one layer of hidden nodes	
