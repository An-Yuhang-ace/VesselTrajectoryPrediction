import tensorflow as tf
import numpy as np
import Model
import csv
from TrajectoryLoader import TrajectoryLoader

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


# network parameters
n_hidden = 128
lstm_step = 6
bp_step = 6
seq_length = 20
batch_size = 300

# test data 
test_loader = TrajectoryLoader()
#test_loader.loadTrajectoryData("TrajectoryMillion.csv")
test_loader.loadTrajectoryData("./DataSet/test.csv")

# restore model
bp_restored = Model.BP(n_hidden, bp_step, batch_size)
checkpoint = tf.train.Checkpoint(BP_network = bp_restored)
checkpoint.restore(tf.train.latest_checkpoint('./SaveBP'))

lstm_restored = Model.LSTM(n_hidden, lstm_step, batch_size)
checkpoint = tf.train.Checkpoint(LSTM_network = lstm_restored)
checkpoint.restore(tf.train.latest_checkpoint('./SaveLSTM'))

test_num = 10

# one-point prediction test

sum_loss = 0.0

for i in range(test_num):
    test_x, test_y = test_loader.getBatchBP(batch_size, bp_step)

    pred = bp_restored(test_x)
    loss = tf.keras.losses.MSE(test_y[:, 1:3], pred[:, 1:3])
    loss = tf.sqrt(loss)
    loss = tf.reduce_mean(loss)
    print("BPtest%d :loss %f" % (i+1, loss.numpy()))
    sum_loss += loss.numpy()
print(sum_loss / 10)


sum_loss = 0.0
for i in range(test_num):
    test_x, test_y = test_loader.getBatchLSTM(batch_size, seq_length)
    test_x = StepProcess(test_x, batch_size, seq_length, lstm_step)
    pred = lstm_restored(test_x)
    loss = tf.keras.losses.MSE(test_y[:, 1:3], pred[:, 1:3])
    loss = tf.sqrt(loss)
    loss = tf.reduce_mean(loss)
    print("LSTMtest%d :loss %f" % (i+1, loss.numpy()))
    sum_loss += loss.numpy()

print(sum_loss / 10)



