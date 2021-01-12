import model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data_utils

data = data_utils.load_training_data("accel_y")

num_epochs = 500

model.reset_session_and_model()
with tf.compat.v1.Session() as sess:
    train_config = model.ModelConfig()
    test_config = model.ModelConfig()
    #train_config.num_layers = 1
    #test_config.num_layers = 1
    test_config.batch_size = 1
    test_config.num_steps = 1
    loader = data_utils.DataLoader(data=data,batch_size=train_config.batch_size, num_steps=train_config.num_steps)
    train_model = model.RNNModel(train_config, True)
    test_model = model.RNNModel(test_config, False)
    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()
    for idx in range(num_epochs):
        epoch_loss = train_model.train_for_epoch(sess, loader)
        if (idx+1) % 100 == 0:
            saver.save(sess, './models/RNNMODELTESTRUNY.ckpt', global_step=idx)
            print(idx, ' ', epoch_loss)
    sample_preds = test_model.predict(sess, seq_len=8000)

    plt.plot(sample_preds)
    plt.show()
