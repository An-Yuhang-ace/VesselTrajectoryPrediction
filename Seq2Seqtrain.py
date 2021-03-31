import tensorflow as tf
import numpy as np
import random
import csv
from TrajectoryLoader import TrajectoryLoader
import Model


# parameters for traning
learnig_rate = 0.001
num_batches = 100000
batch_size = 256
display_step = 50
# parameters for seq2seq model
n_lstm = 128
encoder_length = 20
decoder_length = 10

# Choose Adam optimizer.
optimizer = tf.keras.optimizers.Adam(learnig_rate)

# Create and build encoder and decoder.
encoder = Model.Encoder(n_lstm, batch_size)
decoder = Model.Decoder(n_lstm, batch_size)

x = np.zeros((batch_size, 1, 5), dtype=np.float32)              
output = encoder(x)
decoder(x, output[1:])
encoder.summary()
decoder.summary()

#tensorboard
summary_writer = tf.summary.create_file_writer('tensorboard')
tf.summary.trace_on(profiler=True)
# checkpoint
checkpoint1 = tf.train.Checkpoint(Encoder = encoder)
manager1 = tf.train.CheckpointManager(checkpoint1, directory = './SaveEncoder', checkpoint_name = 'Encoder.ckpt', max_to_keep = 10)
checkpoint2 = tf.train.Checkpoint(Decoder = decoder)
manager2 = tf.train.CheckpointManager(checkpoint2, directory = './SaveDecoder', checkpoint_name = 'Decoder.ckpt', max_to_keep = 10)


def RunOptimization(source_seq, target_seq_in, target_seq_out, step):
    loss = 0
    decoder_length = target_seq_out.shape[1]
    with tf.GradientTape() as tape:
        encoder_outputs = encoder(source_seq)
        states = encoder_outputs[1:]
        y_sample = 0
        for t in range(decoder_length):
            '''
            if t == 0 or random.randint(0,1) == 0:
                decoder_in = tf.expand_dims(target_seq_in[:, t], 1)
            else:
                decoder_in = tf.expand_dims(y_sample, 1)
            '''
            decoder_in = tf.expand_dims(target_seq_in[:, t], 1)
            logit, de_state_h, de_state_c= decoder(decoder_in, states)
            y_sample = logit
            states = de_state_h, de_state_c
            # loss function : RSME TODO
            loss_0 = tf.keras.losses.MSE(target_seq_out[:, t, 1:3], logit[:, 1:3])
            loss += tf.sqrt(loss_0)# TODO
        
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables)) 
     
    
    loss = tf.reduce_mean(loss)  
    loss = loss / decoder_length
    with summary_writer.as_default():
        tf.summary.scalar("loss", loss.numpy(), step = step)   

    return loss

# Load trajectory data.
seq2seq_loader = TrajectoryLoader()
seq2seq_loader.loadTrajectoryData("./DataSet/TrajectoryMillion.csv")

for batch_index in range(1, num_batches+1):
    seq_encoder, seq_decoder = seq2seq_loader.getBatchSeq2Seq(batch_size, encoder_length, decoder_length)
    seq_decoder_in = seq_decoder[:, :decoder_length, :]
    seq_decoder_out = seq_decoder[:, 1:decoder_length+1, :]
    loss = RunOptimization(seq_encoder, seq_decoder_in, seq_decoder_out, batch_index)

    if batch_index % display_step == 0:
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
        path1 = manager1.save(checkpoint_number = batch_index)
        path2 = manager2.save(checkpoint_number = batch_index)


with summary_writer.as_default():
    tf.summary.trace_export(name = "model_trace", step = 0, profiler_outdir = 'tensorboard')
