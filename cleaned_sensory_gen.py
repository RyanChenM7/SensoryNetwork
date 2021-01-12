import model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

from random import randint
from random import shuffle
from random import random
import time

folder_path = 'finished_models/Accelerometer/'

ckpt_path_x = folder_path + 'AccelXPhase2.ckpt-2499'
ckpt_path_y = folder_path + 'AccelXPhase2.ckpt-2499'
ckpt_path_z = folder_path + 'AccelXPhase2.ckpt-2499'

ckpt_all = [ckpt_path_x, ckpt_path_y, ckpt_path_z]

seq_len = 60


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


(test_x, sess_x), (test_y, sess_y), (test_z, sess_z) = [load_model(ckpt) for ckpt in ckpt_all]

names = [i&1 for i in range(seq_len)]

with open("data/" + r"accelgyro_xyz.txt", "a+") as file:
    time1 = time.time()
    for i in range(1, 100000+1):

        fake_xyz = [np.asarray(test_x.predict(sess_x, seq_len))*0.75-0.375,
                    np.asarray(test_y.predict(sess_y, seq_len))*0.75-0.375,
                    np.asarray(test_z.predict(sess_z, seq_len))*0.75-0.375]

        shuffle(fake_xyz)

        file.write(str(list(zip(names, *fake_xyz))))
        file.write('\n')

        if (i+1) % 100 == 0:
            print("write 100 lines:", time.time()-time1)
            time1 = time.time()





