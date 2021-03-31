import csv
import tensorflow as tf
import numpy as np
from TrajectoryLoader import TrajectoryLoader
import Model

# parameters for mini-batch gradient descent training
learning_rate = 0.001
training_step = 30000
batch_size = 1024
display_step = 100

# parameters for network model
n_hidden = 128
bp_step = 6

# Load trajectory data.
bp_loader = TrajectoryLoader()
bp_loader.loadTrajectoryData("./DataSet/TrajectoryMillion.csv")

# Create bp model and build it.
neural_net = Model.BP(n_hidden, bp_step, batch_size)

x = np.zeros((batch_size, bp_step, 5), dtype= np.float32)
neural_net(x)
neural_net.summary()

# Choose Adam optimizer.
optimizer = tf.optimizers.Adam(learning_rate)

#tensorboard
summary_writer = tf.summary.create_file_writer('tensorboard')
tf.summary.trace_on(profiler=True)

# checkpoint
checkpoint = tf.train.Checkpoint(BP_network = neural_net)
manager = tf.train.CheckpointManager(checkpoint, directory = './SaveBP', checkpoint_name = 'BPnetwork.ckpt', max_to_keep = 10)

# optimazation process.
for batch_index in range(1, training_step + 1):
    x, y = bp_loader.getBatchBP(batch_size, bp_step)
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



