import csv
import tensorflow as tf
import numpy as np
from TrajectoryLoader import TrajectoryLoader
import Model

# parameters for mini-batch gradient descent training
learning_rate = 0.0005
num_batches = 5000
batch_size = 1024
display_step = 100

# parameters for LSTM network
n_hidden = 128
lstm_step = 6
seq_length = 40

lstm_loader = TrajectoryLoader()
lstm_loader.loadTrajectoryData("./DataSet/TrajectoryMillion.csv")

neural_net = Model.LSTMFixed(n_hidden, seq_length, lstm_step, batch_size)

# Specifying the input shape in advance
x = np.zeros((batch_size, seq_length, 5), dtype= np.float32)
neural_net(x)

neural_net.summary()

# choose Adam optimizer.
optimizer = tf.optimizers.Adam(learning_rate)
#optimizer = tf.optimizers.SGD(learning_rate)

#tensorboard
summary_writer = tf.summary.create_file_writer('tensorboard')
tf.summary.trace_on(profiler=True)

# checkpoint
checkpoint = tf.train.Checkpoint(LSTM_network = neural_net)
manager = tf.train.CheckpointManager(checkpoint, directory = './SaveLSTM', checkpoint_name = 'LSTMnetwork.ckpt', max_to_keep = 10)

# optimazation process.
for batch_index in range(1, num_batches+1):
    # x.shape: [batch_size, seq_length, 5], y.shape: [batch_size, 5]
    x, y = lstm_loader.getBatchLSTM(batch_size, seq_length)
    with tf.GradientTape() as tape:
        pred = neural_net(x)
        loss = tf.keras.losses.MSE(y, pred)
        loss = tf.reduce_mean(loss)
        with summary_writer.as_default():
            tf.summary.scalar("loss", loss.numpy(), step = batch_index)
    gradients = tape.gradient(loss, neural_net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, neural_net.trainable_variables))

    if batch_index % display_step == 0:
        pred = neural_net(x)
        loss = tf.keras.losses.MSE(y, pred)
        loss = tf.reduce_mean(loss)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
        path = manager.save(checkpoint_number = batch_index)

with summary_writer.as_default():
    tf.summary.trace_export(name = "model_trace", step = 0, profiler_outdir = 'tensorboard')




