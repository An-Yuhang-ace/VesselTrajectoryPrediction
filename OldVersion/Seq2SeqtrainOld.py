import tensorflow as tf
import numpy as np
import random
import csv
from TrajectoryLoader import TrajectoryLoader
import Model


# parameters for traning
learnig_rate = 0.0005
num_batches = 5000
batch_size = 1024
display_step = 100

# parameters for seq2seq model
n_lstm = 128
encoder_length = 40
decoder_length = 20

# Data loader
seq2seq_loader = TrajectoryLoader()
seq2seq_loader.loadTrajectoryData("./DataSet/TrajectoryMillion.csv")

# Build encoder and decoder
encoder = Model.Encoder(n_lstm, batch_size)
decoder = Model.Decoder_LSTMCell(n_lstm, batch_size)

# Specifying the input shape in advance
x = np.zeros((1024, 40, 5), dtype= np.float32)
encoder(x)
states = decoder.lstm.get_initial_state(batch_size = batch_size, dtype = np.float32)
decoder(x, 40, states, is_training = True)

encoder.summary()
decoder.summary()

# choose Adam optimizer.
optimizer = tf.keras.optimizers.Adam(learnig_rate)

#tensorboard
summary_writer = tf.summary.create_file_writer('tensorboard')
tf.summary.trace_on(profiler=True)

# checkpoint
checkpoint1 = tf.train.Checkpoint(Encoder = encoder)
manager1 = tf.train.CheckpointManager(checkpoint1, directory = './SaveEncoder', checkpoint_name = 'Encoder.ckpt', max_to_keep = 10)
checkpoint2 = tf.train.Checkpoint(Decoder = decoder)
manager2 = tf.train.CheckpointManager(checkpoint2, directory = './SaveDecoder', checkpoint_name = 'Decoder.ckpt', max_to_keep = 10)


def RunOptimization(source_seq, target_seq_in, step):
    with tf.GradientTape() as tape:
        en_outputs = encoder(source_seq)
        en_states = en_outputs[1:]

        de_states = en_states
        #de_states = decoder.lstm.get_initial_state(batch_size = batch_size, dtype = tf.float32)
        pred = decoder(target_seq_in, decoder_length, de_states)
        # pred序列对应的实际值，需要去掉target_seq_in的第一位。
        loss = tf.keras.losses.MSE(target_seq_in[:, 1:, :], pred)
        loss = tf.reduce_mean(loss)
        with summary_writer.as_default():
            tf.summary.scalar("loss", loss, step = step)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss

for batch_index in range(1, num_batches+1):
    seq_encoder, seq_decoder = seq2seq_loader.getBatchSeq2Seq(batch_size, encoder_length, decoder_length)
    loss = RunOptimization(seq_encoder, seq_decoder, batch_index)

    if batch_index % display_step == 0:
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
        path1 = manager1.save(checkpoint_number = batch_index)
        path2 = manager2.save(checkpoint_number = batch_index)


with summary_writer.as_default():
    tf.summary.trace_export(name = "model_trace", step = 0, profiler_outdir = 'tensorboard')
