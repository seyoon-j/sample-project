# revision by sy 23.07.31

import matplotlib.pyplot as plt

from network import Network
from loadData import *
from network import Network
import os
import time
import pickle
import pandas as pd

origin_folder = os.getcwd()
# network_folder = r"H:\내 드라이브\Compact Model\MSK\NMDL_AUX\DualGate\simulation_results\2023-7-25_15-45-47\run_12"
network_folder = r"C:\Users\yoony\Documents\1.NMDL\VS_code\simulation_results\2023-7-31_13-15-28\run_1"
os.chdir(network_folder)
with open('network.pkl', 'rb') as file:
    net = pickle.load(file)

os.chdir(origin_folder)
# data_filename = r'H:\내 드라이브\Compact Model\기초 교육\Run\MSK_testrun\DG_testdata2.xlsx'
data_filename = r'C:\Users\yoony\Documents\1.NMDL\VS_code\DG_trainingdata.xlsx'

input_whole, output_whole = encoded_data(data_filename)
test_data, valid_data = wrap_data(input_whole, output_whole, training_ratio=1)
validation_results = [[x, net.feedforward(x), y] for x, y in test_data]

vtg = []
idd_ff = []
idd_answer = []
for i in range(len(validation_results)):
    temp_input = validation_results[i][0]
    temp_feedforward = validation_results[i][1]
    temp_measured = validation_results[i][2]
    vtg.append(temp_input[0])
    idd_ff.append((np.power(10, temp_feedforward[0])-1)*1E-9)
    idd_answer.append((np.power(10, temp_measured[0])-1)*1E-9)

plt.plot(vtg, idd_answer, 'ok')
# plt.plot(vtg, idd_ff, 'sr')
plt.xlabel('VTG')
plt.ylabel('IDD')
# plt.yscale('log')
plt.show()

