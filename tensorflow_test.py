import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from os import listdir, getcwd
from os.path import isfile, join

import data_utils


def minmax_scale(arr):
    _min = min(arr)
    _max = max(arr)
    scaling = (_max - _min)
    f = lambda x : (x - _min)/scaling
    return list(map(f, arr))


'''
accel_training = []
gyro_training = []

current_path = getcwd()
training_path = join(current_path, "training_data")

only_files = [join(training_path, f) for f in listdir(training_path) if isfile(join(training_path, f))]

for file in only_files:
    flag = False
    time_init = 0
    training_data = {'accel': [], 'gyro': []}
    with open(file, "r") as f:
        for line in f:
            s = eval(json.loads(line.rstrip()))
            values = list(s.values())[1:-1]

            time_stamp = int(s['ts'])
            if not flag:
                flag = True
                time_init = time_stamp
            training_data[s['type']].append(tuple(values + [(time_stamp - time_init)/1e9]))

        f.close()
    accel_training.append(np.asarray(training_data['accel'][:300]))
    gyro_training.append(np.asarray(training_data['gyro'][:300]))


def sigmoid(x):
    return 1/(1 + np.exp(-x))


start = 5



def run_visual():
    fig, a = plt.subplots(N, 3)
    a[0][0].set_title('x vs time')
    a[0][1].set_title('y vs time')
    a[0][2].set_title('z vs time')
    for i in range(N):
        case = accel_training[start+i]
        xdat, ydat, zdat, tsdat = zip(*case)

        a[i][0].plot(range(len(xdat)), xdat)
        a[i][1].plot(range(len(ydat)), ydat)
        a[i][2].plot(range(len(zdat)), zdat)

        #a[i][0].plot(tsdat, xdat)
        #a[i][1].plot(tsdat, ydat)
        #a[i][2].plot(tsdat, zdat)


for i in range(len(accel_training)):
    case = accel_training[i]
    xdat, ydat, zdat, tsdat = zip(*case)
    #print(list(zip(*case)))
'''

# number of test cases on the graph
N = 8

x_data = data_utils.load_training_data("touch_y")
fig, a = plt.subplots(N, 5)
a[0][0].set_title('')
a[0][1].set_title('')
a[0][2].set_title('')


for i in range(5):
    for j in range(N):
        dat = x_data[5*j + i]
        #dat = minmax_scale(dat)
        a[j][i].plot(range(500), dat)

plt.show()

