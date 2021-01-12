import model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

from random import randint
from random import shuffle
from random import random
import time

# ckpt_path = 'models/' + 'TimePhase1.ckpt-699'

folder_patha = 'finished_models/Accelerometer/'
folder_pathg = 'finished_models/Gyro/'

ckpt_path_xa = folder_patha + 'AccelXPhase2.ckpt-2499'
ckpt_path_ya = folder_patha + 'AccelXPhase2.ckpt-2499'
ckpt_path_za = folder_patha + 'AccelXPhase2.ckpt-2499'

ckpt_path_xg = folder_pathg + 'GyroXPhase2.ckpt-2499'
ckpt_path_yg = folder_pathg + 'GyroXPhase2.ckpt-2499'
ckpt_path_zg = folder_pathg + 'GyroXPhase2.ckpt-2499'

ckpt_all_a = [ckpt_path_xa, ckpt_path_ya, ckpt_path_za]
ckpt_all_g = [ckpt_path_xg, ckpt_path_yg, ckpt_path_zg]

seq_len = 30


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

    #print_tensors_in_checkpoint_file(file_name=ckpt_path, tensor_name='', all_tensors=False)

    saver = tf.compat.v1.train.Saver()
    sess.run(tf.compat.v1.global_variables_initializer())
    saver.restore(sess, ckpt_path)

    _ = test_model.predict(sess)

    return test_model, sess


(test_xa, sess_xa), (test_ya, sess_ya), (test_za, sess_za) = [load_model(ckpt) for ckpt in ckpt_all_a]

(test_xg, sess_xg), (test_yg, sess_yg), (test_zg, sess_zg) = [load_model(ckpt) for ckpt in ckpt_all_g]


def add(num):
    return lambda x: x*random()*1.5 + num


s3 = add(3.2)
s5 = add(5.2)
s7 = add(7.2)


def gen_ts(seq_len):
    ts = 0
    arr = []
    for i in range(seq_len):
        arr.append(ts)
        ts += randint(9900000, 10000000)

    return arr


names_a = [0 for i in range(30)]
names_g = [1 for i in range(30)]

'''
with open("file.txt", "a+") as file:
    time1 = time.time()
    for i in range(10000):
        fake_xyz = [np.asarray(test_x.predict(sess_x, seq_len))*random()*1.5 + 3,
                    np.asarray(test_y.predict(sess_y, seq_len))*random()*1.5 + 5,
                    np.asarray(test_z.predict(sess_z, seq_len))*random()*1.5 + 7]
            
        fake_xyz = [list(map(s3, test_x.predict(sess_x, seq_len))),
                    list(map(s5, test_y.predict(sess_y, seq_len))),
                    list(map(s7, test_z.predict(sess_z, seq_len)))]
        
        shuffle(fake_xyz)

        timestamps = gen_ts()

        file.write(str(list(zip(names, *fake_xyz, timestamps))))
        file.write('\n')

        if i % 100 == 0:
            print(time.time() - time1)
            time1 = time.time()
'''


def get_accel_data():
    accel = [np.asarray(test_xa.predict(sess_xa, seq_len)),
             np.asarray(test_ya.predict(sess_ya, seq_len)),
             np.asarray(test_za.predict(sess_za, seq_len))]

    accel[0] = accel[0] + random()*1.5 + 1
    accel[1] = accel[1] + random()*3
    accel[2] = accel[2] + 8.5

    magnitude = (accel[0][0]**2 + accel[1][0]**2 + accel[2][0]**2)**0.5
    scaling = (9.8 + random()*0.5)/magnitude
    for arr in accel:
        arr *= scaling

    return accel


def get_gyro_data():
    gyro = [np.asarray(test_xg.predict(sess_xg, seq_len)),
            np.asarray(test_yg.predict(sess_yg, seq_len)),
            np.asarray(test_zg.predict(sess_zg, seq_len))]

    gyro[0] = (gyro[0] - 0.5)*random()
    gyro[1] = (gyro[1] - 0.5)*random()
    gyro[2] = (gyro[2] - 0.5)*random()

    return gyro


with open("data/" + r"accelgyro_xyz.txt", "a+") as file:
    time1 = time.time()
    for i in range(1, 100000 + 1):

        accel = get_accel_data()

        gyro = get_gyro_data()

        # shuffle(accel)

        # 30 accel followed by 30 gyro
        accel = list(zip(names_a, *accel))
        gyro = list(zip(names_g, *gyro))

        # interlace 30 accel with 30 gyro
        fake_xyz = [val for pair in zip(accel, gyro) for val in pair]

        file.write(str(fake_xyz))
        file.write('\n')

        if (i+1) % 20 == 0:
            print("write 20 lines:", time.time() - time1)
            time1 = time.time()
            print(fake_xyz)





