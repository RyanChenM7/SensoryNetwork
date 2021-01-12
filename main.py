import model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data_utils
from random import randint

import time

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


# data_utils.write_dimensional_data()


data = data_utils.load_training_data("touch_y")


# folder_path = 'finished_models/Accelerometer/'
folder_path = 'finished_models/Touch/'

ckpt_path = folder_path + 'TouchY.ckpt'
# ckpt_path = 'models/' + '!AccelX.ckpt-999'
seq_len = 500


model.reset_session_and_model()
tf.compat.v1.reset_default_graph()


with tf.compat.v1.Session() as sess:
    test_config = model.ModelConfig()
    test_config.num_layers = 1
    test_config.batch_size = 1
    test_config.num_steps = 1
    test_model = model.MDNModel(test_config, True)
    test_model.is_training = False
    sess.run(tf.compat.v1.global_variables_initializer())
    # print(tf.compat.v1.global_variables())
    saver = tf.compat.v1.train.Saver()
    # print_tensors_in_checkpoint_file(file_name=ckpt_path, tensor_name='', all_tensors=False)
    saver.restore(sess, ckpt_path)

    t1 = time.time()*1000
    for i in range(100):

        fake_data = test_model.predict(sess, seq_len)

        true_data = data[randint(0, 29), :seq_len]
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        axes[0].plot(true_data)
        axes[0].set_title('True data')
        axes[1].plot(fake_data)
        axes[1].set_title('Fake data')

        plt.setp(axes, yticks=[x/10.0 - 1.5 for x in range(40)])
        print(fake_data)

        plt.show()


print(time.time()*1000-t1)


