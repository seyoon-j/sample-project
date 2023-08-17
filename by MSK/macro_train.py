# revision by sy 23.07.31

from network import Network
from loadData import *
from folder_creator import *
import os
import time

maindir = os.getcwd()
os.chdir(f"{maindir}/simulation_results")

# training_filename = r'H:\내 드라이브\Compact Model\기초 교육\Run\MSK_testrun\DG_trainingdata.xlsx'
training_filename = r'C:\Users\yoony\Documents\1.NMDL\VS_code\DG_trainingdata.xlsx'
input_whole, output_whole = encoded_data(training_filename)
training_data, valid_data = wrap_data(input_whole, output_whole, training_ratio=0.7)
test_data = training_data
"""macro = [network_size, epoch, mini_batch_size, learning_rate]"""

macro = [[[3, 5, 15, 4], 100, 100, 0.05],
        #  [[3, 10, 15, 4], 2000, 100, 0.05],
        #  [[3, 15, 15, 4], 2000, 100, 0.05],
        #  [[3, 20, 15, 4], 2000, 100, 0.05],
        #  [[3, 30, 15, 4], 2000, 100, 0.05],
        #  [[3, 15, 5, 4], 2000, 100, 0.05],
        #  [[3, 15, 10, 4], 2000, 100, 0.05],
        #  [[3, 15, 15, 4], 2000, 100, 0.05],
        #  [[3, 15, 20, 4], 2000, 100, 0.05],
        #  [[3, 15, 30, 4], 2000, 100, 0.05],
        #  [[3, 30, 30, 4], 2000, 100, 0.05],
         [[3, 50, 50, 4], 2000, 100, 0.05]]
# Create folder name with date
mainfolder()

for i in range(len(macro)):
    print(f"Training {i + 1} / {len(macro)}")
    net_size    = macro[i][0]
    epoch       = macro[i][1]
    batch_size  = macro[i][2]
    eta         = macro[i][3]
    net = Network(net_size)
    start_time = time.time()
    net.SGD(training_data, epoch, batch_size, eta, test_data=test_data)
    end_time = time.time()
    training_time = end_time - start_time
    net.train_time = training_time
    # Change directory to save directory
    subfolders(i + 1, net, "network")
    print(f"Training took {net.train_time} seconds")
    print(f"\n======================")

os.chdir(maindir)