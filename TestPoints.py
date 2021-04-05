import tensorflow as tf
import numpy as np
import Model
import csv
from TestLoader import TestLoader


# network parameters
n_lstm = 128
lstm_step = 6
batch_size = 512

attention_func1 = 'dot' 
attention_func2 = 'general' 
attention_func3 = 'concat'

# Restore LSTM model, seq2seq model and attention-seq2seq model.
lstm_restored = Model.LSTM(n_lstm, lstm_step, batch_size) 
checkpoint1 = tf.train.Checkpoint(LSTM_network = lstm_restored)
checkpoint1.restore(tf.train.latest_checkpoint('./SaveLSTM'))


encoder = Model.Encoder(n_lstm, batch_size)
checkpoint2 = tf.train.Checkpoint(Encoder = encoder)
checkpoint2.restore(tf.train.latest_checkpoint('./SaveEncoder'))

decoder = Model.Decoder(n_lstm, batch_size)
checkpoint3 = tf.train.Checkpoint(Decoder = decoder)
checkpoint3.restore(tf.train.latest_checkpoint('./SaveDecoder'))


encoder_a = Model.Encoder(n_lstm, batch_size)
checkpoint4 = tf.train.Checkpoint(EncoderAttention = encoder_a)
checkpoint4.restore(tf.train.latest_checkpoint('./SaveEncoderAttention'))

decoder_a = Model.DecoderAttention(n_lstm, batch_size, attention_func2)
checkpoint5 = tf.train.Checkpoint(DecoderAttention = decoder_a)
checkpoint5.restore(tf.train.latest_checkpoint('./SaveDecoderAttention'))

# Test functions for models: TestSeq2Seq, TestSeq2SeqAttention, TestLSTM.

def TestSeq2Seq(source_seq, target_seq_in, target_seq_out):
    """Test restored seq2seq model.

    Args:
        source_seq (shape of [batch_size, source_length, 5])
        target_seq_in (shape of [batch_size, pred_length, 5])
        target_seq_out (shape of [batch_size, pred_length, 5])

    Returns:
        loss [tensor]: Root Mean Squre Error loss of prediction of points.
    """    
    loss = 0
    pred = []
    decoder_length = target_seq_out.shape[1]
    # Encode the source.
    encoder_outputs = encoder(source_seq)
    states = encoder_outputs[1:]
    # Decoder predicts the target_seq.
    decoder_in = tf.expand_dims(target_seq_in[:, 0], 1)
    for t in range(decoder_length):
        logit, de_state_h, de_state_c= decoder(decoder_in, states)
        decoder_in = tf.expand_dims(logit, 1)
        states = de_state_h, de_state_c
        # loss function : RSME TODO
        loss_0 = tf.keras.losses.MSE(target_seq_out[:, t, 1:3], logit[:, 1:3])
        loss += tf.sqrt(loss_0)# TODO
        
    loss = tf.reduce_mean(loss)  
    loss = loss / decoder_length
    return loss


def TestAttentionSeq2SeqOld(source_seq, target_seq_in, target_seq_out):
    """Test restored attention_seq2seq model.

    Args:
        source_seq (shape of [batch_size, source_length, 5])
        target_seq_in (shape of [batch_size, pred_length, 5])
        target_seq_out (shape of [batch_size, pred_length, 5])

    Returns:
        loss [tensor]: Root Mean Squre Error loss of prediction of points.
    """    
    loss = 0
    decoder_length = target_seq_out.shape[1]
    # Encode the source.
    encoder_outputs = encoder_a(source_seq)
    states = encoder_outputs[1:]
    # Decoder predicts the target_seq.
    decoder_in = tf.expand_dims(target_seq_in[:, 0], 1)
    for t in range(decoder_length):
        logit, _, de_state_h, de_state_c, _= decoder_a(decoder_in, states, encoder_outputs[0])
        decoder_in = tf.expand_dims(logit, 1)
        states = de_state_h, de_state_c
        # loss function : RSME TODO
        loss_0 = tf.keras.losses.MSE(target_seq_out[:, t, 1:3], logit[:, 1:3])
        loss += tf.sqrt(loss_0)# TODO
        
    loss = tf.reduce_mean(loss)  
    loss = loss / decoder_length
    return loss

def TestAttentionSeq2Seq(source_seq, target_seq_in, target_seq_out):
    """Test restored attention_seq2seq model.

    Args:
        source_seq (shape of [batch_size, source_length, 5])
        target_seq_in (shape of [batch_size, pred_length, 5])
        target_seq_out (shape of [batch_size, pred_length, 5])

    Returns:
        loss [tensor]: Root Mean Squre Error loss of prediction of points.
    """    
    loss = 0
    decoder_length = target_seq_out.shape[1]
    # Encode the source.
    encoder_outputs = encoder_a(source_seq)
    states = encoder_outputs[1:]
    history = encoder_outputs[0]
    # Decoder predicts the target_seq.
    decoder_in = tf.expand_dims(target_seq_in[:, 0], 1)
    for t in range(decoder_length):
        logit, lstm_out, de_state_h, de_state_c, _= decoder_a(decoder_in, states, history)
        decoder_in = tf.expand_dims(logit, 1)
        history_new = tf.expand_dims(lstm_out, 1)
        history = tf.concat([history[:, 1:], history_new], 1)
        states = de_state_h, de_state_c
        # loss function : RSME TODO
        loss_0 = tf.keras.losses.MSE(target_seq_out[:, t, 1:3], logit[:, 1:3])
        loss += tf.sqrt(loss_0)# TODO
        
    loss = tf.reduce_mean(loss)  
    loss = loss / decoder_length
    return loss

def TestLSTM(test_x, test_y):
    """Test restored lstm model.

    Args:
        test_x (shape of [batch_size, source_length, 5])
        test_y ([batch_size, pred_length, 5])

    Returns:
        loss [tensor]: Root Mean Squre Error loss of prediction of points.
    """    
    loss = 0.0
    seq_length = test_y.shape[1]
    for t in range(seq_length):
        lstm_in = StepProcess(test_x, batch_size, source_length, lstm_step)
        logit = lstm_restored(lstm_in)
        # loss function : RSME TODO
        loss_0 = tf.keras.losses.MSE(test_y[:, t, 1:3], logit[:, 1:3])
        loss += tf.sqrt(loss_0)# TODO
        pred_point = np.reshape(logit.numpy(), [batch_size, 1, 5])
        test_x = np.concatenate((test_x[:, 1:source_length, :], pred_point), axis=1) 
    
    loss = tf.reduce_mean(loss)
    loss = loss / seq_length
    return loss

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


# Load test data
test_loader = TestLoader()
test_loader.loadTestTrajectory("./DataSet/test_fix.csv")

source_length = 120
target_testset = [20, 40, 60, 80, 100, 120]

for target_length in target_testset:
    source_seq, source_coordinates, target_seq, target_coordinates= test_loader.getTestSeq2Seq(batch_size, source_length, target_length)

    # LSTM
    test_x = source_seq
    test_y = target_seq[:, 1:target_length+1, :]
    loss = TestLSTM(test_x, test_y)
    print("Result of LSTM_%d: %f" % (target_length, loss.numpy()))

    # Seq2Seq
    target_seq_in = target_seq[:, :target_length, :]
    target_seq_out = target_seq[:, 1:target_length+1, :]

    loss = TestSeq2Seq(source_seq, target_seq_in, target_seq_out)
    print("Result of Seq2Seq_%d: %f" % (target_length, loss.numpy()))

    # Seq2SeqAttention
    loss = TestAttentionSeq2Seq(source_seq, target_seq_in, target_seq_out)
    print("Result of Seq2SeqAttention_%d: %f" % (target_length, loss.numpy()))
