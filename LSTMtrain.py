import csv
import tensorflow as tf
import numpy as np
from TrajectoryLoader import TrajectoryLoader
import Model

# parameters for mini-batch gradient descent training
learning_rate = 0.0005
num_batches = 10000
batch_size = 256
display_step = 100
# parameters for LSTM network
n_lstm = 128
lstm_step = 6
seq_length = 20

# Choose Adam optimizer.
optimizer = tf.optimizers.Adam(learning_rate)
#optimizer = tf.optimizers.SGD(learning_rate)

# Create lstm model and build it.
neural_net = Model.LSTM(n_lstm, lstm_step, batch_size)

x = np.zeros((batch_size, seq_length-lstm_step, lstm_step*5), dtype= np.float32)
neural_net(x)
neural_net.summary()

#tensorboard
summary_writer = tf.summary.create_file_writer('tensorboard')
tf.summary.trace_on(profiler=True)
# checkpoint
checkpoint = tf.train.Checkpoint(LSTM_network = neural_net)
manager = tf.train.CheckpointManager(checkpoint, directory = './SaveLSTM', checkpoint_name = 'LSTMnetwork.ckpt', max_to_keep = 10)


def StepProcess(input, batch_size, seq_length, lstm_step):
    if lstm_step == 1:
        return input
    else:
        seq_length_new = seq_length - lstm_step + 1
        output = []
        for i in range(lstm_step):
            seq_sub = input[:, i:seq_length_new+i, :]
            output.append(seq_sub)
        return np.concatenate(output, axis=2)

def RunOptimization(x, y, step):
    with tf.GradientTape() as tape:
        pred = neural_net(x)
        # loss function : RMSE TODO
        loss = tf.keras.losses.MSE(y[:, 1:3], pred[:, 1:3])
        loss = tf.sqrt(loss)
    
    gradients = tape.gradient(loss, neural_net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, neural_net.trainable_variables))

    loss = tf.reduce_mean(loss)
    with summary_writer.as_default():
        tf.summary.scalar("loss", loss.numpy(), step = step)
    return loss


# Load trajectory data.
lstm_loader = TrajectoryLoader()
lstm_loader.loadTrajectoryData("./DataSet/TrajectoryMillion.csv")

# optimazation process.
for batch_index in range(1, num_batches+1):
    # x.shape: [batch_size, seq_length, 5], y.shape: [batch_size, 5]
    x, y = lstm_loader.getBatchLSTM(batch_size, seq_length)
    # after StepProcess, x.shape: [batch_size, seq_lenght-lstm_step+1, lstm_step*5]
    x = StepProcess(x, batch_size, seq_length, lstm_step)
    loss = RunOptimization(x, y, batch_index)

    if batch_index % display_step == 0:
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
        path = manager.save(checkpoint_number = batch_index)

with summary_writer.as_default():
    tf.summary.trace_export(name = "model_trace", step = 0, profiler_outdir = 'tensorboard')




