import model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

from random import randint
from random import shuffle
from random import random
import time


folder_path = 'finished_models/Touch/'

ckpt_path_x = folder_path + 'TouchX.ckpt'
ckpt_path_y = folder_path + 'TouchY.ckpt'

ckpt_all = [ckpt_path_x, ckpt_path_y]

seq_len = 75


def load_model(ckpt_path):
    model.reset_session_and_model()
    tf.compat.v1.reset_default_graph()
    test_config = model.ModelConfig()
    test_config.num_layers = 1
    test_config.batch_size = 1
    test_config.num_steps = 1

    sess = tf.compat.v1.Session()
    test_model = model.MDNModel(test_config, True)

    test_model.is_training = False

    # print_tensors_in_checkpoint_file(file_name=ckpt_path, tensor_name='', all_tensors=False)

    saver = tf.compat.v1.train.Saver()
    sess.run(tf.compat.v1.global_variables_initializer())
    saver.restore(sess, ckpt_path)

    _ = test_model.predict(sess)

    return test_model, sess


(test_x, sess_x), (test_y, sess_y) = [load_model(ckpt) for ckpt in ckpt_all]


def scale_x(arr):
    arr = np.asarray(arr)
    intv = (200, 600)
    intv_size = intv[1] - intv[0]
    arr_size = max(arr) - min(arr)
    scaling = intv_size/arr_size
    arr = arr*scaling + intv[0]

    return list(map(int, arr))


def scale_y(arr):
    arr = np.asarray(arr)
    intv = (400, 1700)
    intv_size = intv[1] - intv[0]
    arr_size = max(arr) - min(arr)
    scaling = intv_size/arr_size

    arr = arr*scaling + intv[0]
    return list(map(int, arr))


def get_touch_data():
    touch_x = test_x.predict(sess_x, seq_len)
    touch_y = test_y.predict(sess_y, seq_len)

    touch_x = scale_x(touch_x)
    touch_y = scale_y(touch_y)

    return list(zip(touch_x, touch_y))


with open("data/" + r"touch_data.txt", "a+") as file:
    time1 = time.time()
    for i in range(1, 100000+1):

        touch = get_touch_data()

        file.write(str(touch))
        file.write('\n')

        if (i+1) % 20 == 0:
            print("write 20 lines:", time.time()-time1)
            time1 = time.time()
            print(touch)





