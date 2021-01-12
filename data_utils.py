import numpy as np
import os
import sys
from scipy.io import loadmat
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from os import listdir, getcwd
from os.path import isfile, join

SEQ_LEN = 0

file_name = "training_data_touch"


def minmax_scale(arr):
    _min = min(arr)
    _max = max(arr)
    scaling = (_max - _min)
    if scaling == 0:
        scaling = 1
    f = lambda x : (x - _min)/scaling
    return list(map(f, arr))


def write_touch_data(name, data_training):
    x, y, ts = data_training
    x = np.asarray(x)
    y = np.asarray(y)
    ts = np.asarray(ts)

    np.savetxt("axis_data/" + name + "_x", x, fmt='%s')
    np.savetxt("axis_data/" + name + "_y", y, fmt='%s')
    np.savetxt("axis_data/"+name+"_time", ts, fmt='%s')


def write_np_data(name, data_training):
    data_x = []
    data_y = []
    data_z = []
    data_t = []

    for i in range(len(data_training)):
        case = data_training[i]
        xdat, ydat, zdat, tdat = zip(*case)
        data_x.append(xdat)
        data_y.append(ydat)
        data_z.append(zdat)
        data_t.append(tdat)

    # Scale data between 0 and 1.
    data_x, data_y, data_z = [list(map(minmax_scale, l)) for l in [data_x, data_y, data_z]]

    data_x, data_y, data_z, data_t = list(map(np.asarray, [data_x, data_y, data_z, data_t]))

    np.savetxt("axis_data/" + name + "_x", data_x, fmt='%s')
    np.savetxt("axis_data/" + name + "_y", data_y, fmt='%s')
    np.savetxt("axis_data/" + name + "_z", data_z, fmt='%s')
    np.savetxt("axis_data/" + name + "_t", data_t, fmt='%s')


# Stores training data in the format [ [ (x,y,z,ts), ... ], ... ]
def write_dimensional_data(name):
    accel_training = []
    gyro_training = []

    current_path = getcwd()

    training_path = join(current_path, file_name)

    only_files = [join(training_path, f) for f in listdir(training_path) if isfile(join(training_path, f))]

    if file_name == "training_data_motion":
        SEQ_LEN = 500
        for file in only_files:
            flag = False
            time_init = 0
            x_init = 0
            y_init = 0
            z_init = 0
            training_data = {'accel': [], 'gyro': []}
            with open(file, "r") as f:
                for line in f:
                    s = eval(json.loads(line.rstrip()))
                    values = list(map(float, list(s.values())[1:-1]))

                    time_stamp = int(s['ts'])
                    if not flag:
                        flag = True
                        time_init = time_stamp
                        x_init = float(s['x'])
                        y_init = float(s['y'])
                        z_init = float(s['z'])
                    training_data[s['type']].append([values[0]-x_init, values[1]-y_init,
                                                     values[2]-z_init]+[(time_stamp-time_init)/1e6])

                f.close()
            accel_training.append(training_data['accel'][:SEQ_LEN])
            gyro_training.append(training_data['gyro'][:SEQ_LEN])

        write_np_data(name, accel_training)

    elif file_name == "training_data_touch":
        SEQ_LEN = 500
        x_training = []
        y_training = []
        scaling = 1.0/500.0

        for file in only_files:
            x_series = []
            y_series = []

            with open(file, "r") as f:
                for line in f:
                    s = json.loads(line.rstrip())
                    x = float(s['x'])*scaling
                    y = float(s['y'])*scaling

                    x_series.append(x)
                    y_series.append(y)

                f.close()
                x_training.append(x_series[:SEQ_LEN])
                y_training.append(y_series[:SEQ_LEN])

        touch_training = [x_training, y_training]

        write_touch_data(name, touch_training)


def load_training_data(filename):
    """ Returns a matrix of training data.
    shape of result = (n_exp, len)
    """
    try:
        data = np.loadtxt("axis_data/" + filename)
    except ValueError:
        with open("axis_data/" + filename, "r") as f:
            data = [np.asarray(json.loads(line.rstrip())) for line in f]

    #print(data)
    return data


class DataLoader(object):
    def __init__(self, data, batch_size=10, num_steps=1):
        self.batch_size = batch_size
        self.n_data, self.seq_len = data.shape
        self._data = data[:self.batch_size, :]

        self.num_steps = num_steps
        self._data = self._data.reshape((self.batch_size, self.seq_len, 1))
        self._reset_pointer()

    def _reset_pointer(self):
        self.pointer = 0

    def reset(self):
        self._reset_pointer()

    def has_next(self):
        return self.pointer+self.num_steps < self.seq_len-1

    def next_batch(self):
        batch_xs = self._data[:, self.pointer:self.pointer+self.num_steps, :]
        batch_ys = self._data[:, self.pointer+1:self.pointer+self.num_steps+1, :]

        self.pointer = self.pointer+self.num_steps
        return batch_xs, batch_ys


write_dimensional_data('time')

