import tensorflow as tf
import numpy as np
import Model
import csv
from TestLoader import TestLoader
import os
import simplekml

# network parameters
n_lstm = 128
lstm_step = 6
batch_size = 1

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

decoder_a = Model.DecoderAttention(n_lstm, batch_size, 'general')
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
        pred [np.array(pred)]: The prediction of points. shape of [seq_length, 5].
        loss [tensor]: Root Mean Squre Error loss of prediction of points.
    """    
    loss = 0
    pred = []
    decoder_length = target_seq_out.shape[1]
    # Encode the source.
    encoder_outputs = encoder(source_seq)
    states = encoder_outputs[1:]
    # Decoder predicts the target_seq. decoder_in shape of [batch_size, 1, 5]
    decoder_in = tf.expand_dims(target_seq_in[:, 0], 1)
    for t in range(decoder_length):
        logit, de_state_h, de_state_c= decoder(decoder_in, states)
        # logit shape of [batch_size, 5]
        decoder_in = tf.expand_dims(logit, 1)
        states = de_state_h, de_state_c
        # loss function : RSME TODO
        loss_0 = tf.keras.losses.MSE(target_seq_out[:, t, 1:3], logit[:, 1:3])
        loss += tf.sqrt(loss_0)# TODO
        pred.append(logit[0, :])
        
    loss = tf.reduce_mean(loss)  
    loss = loss / decoder_length
    pred = np.array(pred)
    return pred, loss.numpy()

'''
def TestAttentionSeq2SeqOld(source_seq, target_seq_in, target_seq_out):
    """Test restored attention_seq2seq model.

    Args:
        source_seq (shape of [batch_size, source_length, 5])
        target_seq_in (shape of [batch_size, pred_length, 5])
        target_seq_out (shape of [batch_size, pred_length, 5])

    Returns:
        pred [np.array(pred)]: The prediction of points. shape of [seq_length, 5].
        loss [tensor]: Root Mean Squre Error loss of prediction of points.
    """    
    loss = 0
    pred = []
    decoder_length = target_seq_out.shape[1]
    # Encode the source.
    encoder_outputs = encoder_a(source_seq)
    states = encoder_outputs[1:]
    # Decoder predicts the target_seq. decoder_in shape of [batch_size, 1, 5]
    decoder_in = tf.expand_dims(target_seq_in[:, 0], 1)
    for t in range(decoder_length):
        logit, _, de_state_h, de_state_c, _= decoder_a(decoder_in, states, encoder_outputs[0])
        # logit shape of [batch_size, 5]
        decoder_in = tf.expand_dims(logit, 1)
        states = de_state_h, de_state_c
        # loss function : RSME TODO
        loss_0 = tf.keras.losses.MSE(target_seq_out[:, t, 1:3], logit[:, 1:3])
        loss += tf.sqrt(loss_0)# TODO
        pred.append(logit[0, :])

    loss = tf.reduce_mean(loss)  
    loss = loss / decoder_length
    pred = np.array(pred)
    return pred, loss.numpy()
'''

def TestAttentionSeq2Seq(source_seq, target_seq_in, target_seq_out):
    """Test restored attention_seq2seq model.

    Args:
        source_seq (shape of [batch_size, source_length, 5])
        target_seq_in (shape of [batch_size, pred_length, 5])
        target_seq_out (shape of [batch_size, pred_length, 5])

    Returns:
        pred [np.array(pred)]: The prediction of points. shape of [seq_length, 5].
        loss [tensor]: Root Mean Squre Error loss of prediction of points.
    """    
    loss = 0
    pred = []
    decoder_length = target_seq_out.shape[1]
    # Encode the source.
    encoder_outputs = encoder_a(source_seq)
    states = encoder_outputs[1:]
    history = encoder_outputs[0]
    # Decoder predicts the target_seq. decoder_in shape of [batch_size, 1, 5]
    decoder_in = tf.expand_dims(target_seq_in[:, 0], 1)
    for t in range(decoder_length):
        logit, lstm_out, de_state_h, de_state_c, _= decoder_a(decoder_in, states, history)
        # logit shape of [batch_size, 5]
        decoder_in = tf.expand_dims(logit, 1)
        history_new = tf.expand_dims(lstm_out, 1)
        history = tf.concat([history[:, 1:], history_new], 1)
        states = de_state_h, de_state_c
        # loss function : RSME
        loss_0 = tf.keras.losses.MSE(target_seq_out[:, t, 1:3], logit[:, 1:3])
        loss += tf.sqrt(loss_0)
        pred.append(logit[0, :])

    loss = tf.reduce_mean(loss)  
    loss = loss / decoder_length
    pred = np.array(pred)
    return pred, loss.numpy()

def TestLSTM(test_x, test_y):
    """Test restored lstm model.

    Args:
        test_x (shape of [batch_size, source_length, 5])
        test_y ([batch_size, pred_length, 5])

    Returns:
        pred [np.array(pred)]: The prediction of points. shape of [seq_length, 5].
        loss [tensor]: Root Mean Squre Error loss of prediction of points.
    """    
    loss = 0.0
    pred = []
    seq_length = test_y.shape[1]
    for t in range(seq_length):
        lstm_in = StepProcess(test_x, batch_size, source_length, lstm_step)
        logit = lstm_restored(lstm_in)
        # loss function : RSME TODO
        loss_0 = tf.keras.losses.MSE(test_y[:, t, 1:3], logit[:, 1:3])
        loss += tf.sqrt(loss_0)# TODO
        pred_point = np.reshape(logit.numpy(), [batch_size, 1, 5])
        test_x = np.concatenate((test_x[:, 1:source_length, :], pred_point), axis=1) 
        pred.append(logit[0,:])
    loss = tf.reduce_mean(loss)
    loss = loss / seq_length
    pred = np.array(pred)
    return pred, loss.numpy()

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
target_length = 60

source_seq, source_coordinates, target_seq, target_coordinates= test_loader.getTestSeq2Seq(batch_size, source_length, target_length)

# LSTM
test_x = source_seq
test_y = target_seq[:, 1:target_length+1, :]
pred_lstm, loss = TestLSTM(test_x, test_y)
print("Result of LSTM_%d: %f" % (target_length, loss))

# Seq2Seq
target_seq_in = target_seq[:, :target_length, :]
target_seq_out = target_seq[:, 1:target_length+1, :]

pred_seq2seq, loss = TestSeq2Seq(source_seq, target_seq_in, target_seq_out)
print("Result of Seq2Seq_%d: %f" % (target_length, loss))

# Seq2SeqAttention
pred_aseq2seq, loss = TestAttentionSeq2Seq(source_seq, target_seq_in, target_seq_out)
print("Result of Seq2SeqAttention_%d: %f" % (target_length, loss))


# Coordinates recovery: denormalize and convert to list.

# source coordinates
lng_source = source_coordinates[0, :, 0]
lat_source = source_coordinates[0, :, 1]
lng_source = lng_source * (test_loader.max_train_data[5] - test_loader.min_train_data[5]) + test_loader.min_train_data[5]
lat_source = lat_source * (test_loader.max_train_data[6] - test_loader.min_train_data[6]) + test_loader.min_train_data[6]
lng_source = lng_source.tolist()
lat_source = lat_source.tolist()

kml = simplekml.Kml(open=1)
lng1 = lng_source[0]
lat1 = lat_source[0]
for i in range(1, len(lng_source)):
    lng2 = lng_source[i]
    lat2 = lat_source[i]
    name = '%d' %i
    linestring = kml.newlinestring(name=name)
    linestring.coords = [(lng1, lat1), (lng2, lat2)]
    lng1 = lng2
    lat1 = lat2
kml.save('./Visualization/source.kml')

# true coordinates
lng_true = target_coordinates[0, :, 0]
lat_true = target_coordinates[0, :, 1]
lng_true = lng_true * (test_loader.max_train_data[5] - test_loader.min_train_data[5]) + test_loader.min_train_data[5]
lat_true = lat_true * (test_loader.max_train_data[6] - test_loader.min_train_data[6]) + test_loader.min_train_data[6]
lng_true = lng_true.tolist()
lat_true = lat_true.tolist()

lng0 = lng_true[0]
lat0 = lat_true[0]

kml = simplekml.Kml(open=1)
lng1 = lng_true[0]
lat1 = lat_true[0]
for i in range(1, len(lng_true)):
    lng2 = lng_true[i]
    lat2 = lat_true[i]
    name = '%d' %i
    linestring = kml.newlinestring(name=name)
    linestring.coords = [(lng1, lat1), (lng2, lat2)]
    lng1 = lng2
    lat1 = lat2
kml.save('./Visualization/true.kml')


# pred coordomates
delta_lng = pred_aseq2seq[:, 1]
delta_lat =pred_aseq2seq[:, 2]
delta_lng = delta_lng * (test_loader.max_train_data[1] - test_loader.min_train_data[1]) + test_loader.min_train_data[1]
delta_lat = delta_lat * (test_loader.max_train_data[2] - test_loader.min_train_data[2]) + test_loader.min_train_data[2]
delta_lng = delta_lng.tolist()
delta_lat = delta_lat.tolist()

# delta_lng, delta_lat to lng, lat
lng0 = lng_true[0]
lat0 = lat_true[0]
lng_pred = []
lat_pred = []
for i in range(len(delta_lng)):
    lng = lng0 + delta_lng[i]
    lat = lat0 + delta_lat[i]
    lng_pred.append(lng)
    lat_pred.append(lat)
    lng0 = lng
    lat0 = lat

kml = simplekml.Kml(open=1)
lng1 = lng_true[0]
lat1 = lat_true[0]
for i in range(len(lng_pred)):
    lng2 = lng_pred[i]
    lat2 = lat_pred[i]
    name = '%d' %i
    linestring = kml.newlinestring(name=name)
    linestring.coords = [(lng1, lat1), (lng2, lat2)]
    lng1 = lng2
    lat1 = lat2
kml.save('./Visualization/pred_a.kml')

# pred coordomates
delta_lng = pred_seq2seq[:, 1]
delta_lat = pred_seq2seq[:, 2]
delta_lng = delta_lng * (test_loader.max_train_data[1] - test_loader.min_train_data[1]) + test_loader.min_train_data[1]
delta_lat = delta_lat * (test_loader.max_train_data[2] - test_loader.min_train_data[2]) + test_loader.min_train_data[2]
delta_lng = delta_lng.tolist()
delta_lat = delta_lat.tolist()

lng0 = lng_true[0]
lat0 = lat_true[0]
lng_pred = []
lat_pred = []
for i in range(len(delta_lng)):
    lng = lng0 + delta_lng[i]
    lat = lat0 + delta_lat[i]
    lng_pred.append(lng)
    lat_pred.append(lat)
    lng0 = lng
    lat0 = lat

kml = simplekml.Kml(open=1)
lng1 = lng_true[0]
lat1 = lat_true[0]
for i in range(len(lng_pred)):
    lng2 = lng_pred[i]
    lat2 = lat_pred[i]
    name = '%d' %i
    linestring = kml.newlinestring(name=name)
    linestring.coords = [(lng1, lat1), (lng2, lat2)]
    lng1 = lng2
    lat1 = lat2
kml.save('./Visualization/pred_seq.kml')