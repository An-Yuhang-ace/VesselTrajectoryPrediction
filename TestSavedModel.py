import tensorflow as tf
import numpy as np
import Model
import csv
from TestLoader import TestLoader
import os
import simplekml


encoder_a = tf.saved_model.load('./SavedModel/Save1')
decoder_a = tf.saved_model.load('./SavedModel/Save2')


@tf.function
def PreAttentionSeq2Seq(source_seq, target_seq_in, decoder_length):
    """Test restored attention_seq2seq model.

    Args:
        source_seq (shape of [batch_size, source_length, 5])
        target_seq_in (shape of [batch_size, pred_length, 5])
        target_seq_out (shape of [batch_size, pred_length, 5])

    Returns:
        pred [np.array(pred)]: The prediction of points. shape of [seq_length, 5].
    """    
    pred = []
    # Encode the source.
    encoder_outputs = encoder_a.call(source_seq)
    states = encoder_outputs[1:]
    state_h = states[0]
    state_c = states[1]
    history = encoder_outputs[0]
    # Decoder predicts the target_seq. decoder_in shape of [batch_size, 1, 5]
    decoder_in = tf.expand_dims(target_seq_in, 1)
    for t in range(decoder_length):
        logit, lstm_out, de_state_h, de_state_c, _= decoder_a.call(decoder_in, state_h, state_c, history)
        # logit shape of [batch_size, 5]
        decoder_in = tf.expand_dims(logit, 1)
        history_new = tf.expand_dims(lstm_out, 1)
        history = tf.concat([history[:, 1:], history_new], 1)
        state_h = de_state_h
        state_c = de_state_c
        pred.append(logit[0, :])
    return pred

# Load test data
test_loader = TestLoader()
test_loader.loadTestTrajectory("./DataSet/test_fix.csv")

source_length = 120
target_length = 60

source_seq, source_coordinates, target_seq, target_coordinates= test_loader.getTestSeq2Seq(1, source_length, target_length)

target_seq_in = target_seq[:, :target_length, :]
target_seq_out = target_seq[:, 1:target_length+1, :]

pred_aseq2seq = PreAttentionSeq2Seq(source_seq, target_seq_in[:, 0], target_length)
pred_aseq2seq = np.array(pred_aseq2seq)

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
delta_lat = pred_aseq2seq[:, 2]
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