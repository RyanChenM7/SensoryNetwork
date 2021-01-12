import model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data_utils

data_utils.write_dimensional_data("touch")


def train_axis_data(data_name, save_name):
    data = data_utils.load_training_data(data_name)

    num_epochs = 2500

    model.reset_session_and_model()
    with tf.compat.v1.Session() as sess:
        train_config = model.ModelConfig()
        test_config = model.ModelConfig()
        train_config.learning_rate = 0.0003
        train_config.num_layers = 1
        test_config.num_layers = 1
        test_config.batch_size = 1
        test_config.num_steps = 1
        loader = data_utils.DataLoader(data=data, batch_size=train_config.batch_size, num_steps=train_config.num_steps)
        train_model = model.MDNModel(train_config, True)
        #test_model = model.MDNModel(test_config, False)
        sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver()
        for idx in range(num_epochs):
            epoch_loss = train_model.train_for_epoch(sess, loader)
            if (idx+1) % 100 == 0:
                print(data_name, idx, ' ', epoch_loss)
                saver.save(sess, './models/' + save_name + '.ckpt', global_step=idx)


train_axis_data("touch_x", "TouchX")
train_axis_data("touch_y", "TouchY")



