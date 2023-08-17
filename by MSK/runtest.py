import numpy as np
from network import Network
from encoder import *
from loadData import *

training_filename = r'H:\내 드라이브\Compact Model\기초 교육\Run\MSK_testrun\DG_trainingdata.xlsx'

input_whole, output_whole = encoded_data(training_filename)
training_data, valid_data = wrap_data(input_whole, output_whole, training_ratio=0.7)

test_data = training_data

net = Network([3, 15, 20, 4])
init_weight = net.weights
init_bias = net.biases


epoch = 500
mini_batch_size = 100
learning_rate = 0.01

net.SGD(training_data, epoch, mini_batch_size, learning_rate, test_data=test_data)

final_weight = net.weights
final_bias = net.biases
