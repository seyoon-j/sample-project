import random
import numpy as np
from network import Network
from encoder import *
from loadData import *


training_filename = r'H:\내 드라이브\Compact Model\기초 교육\Run\MSK_testrun\DG_trainingdata.xlsx'

input_whole, output_whole = encoded_data(training_filename)
test_data, valid_data = wrap_data(input_whole, output_whole, training_ratio=0.7)

training_data = test_data
eta = 0.01
mini_batch_size = 5
n = len(training_data)
tempvar = training_data[0 : 5]
#mixed_training_data = tuple(random.sample(training_data, len(training_data)))
# shuffle whole training data, separate into "mini_batch_size" number of mini batches
#mini_batches = [training_data[k: k + mini_batch_size] for k in range(0, n, mini_batch_size)]
#testtuple = mini_batches[0]
#
#counter = 0
#for mini_batch in testtuple:
#    print(type(mini_batch))
#    print(len(mini_batch))
#    for x, y in mini_batch:
#        print(x)
#        print(y)
#    break


