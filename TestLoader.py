import tensorflow as tf
import numpy as np
import Model
import csv

class TestLoader():
    def __init__(self):
        super().__init__()

    def row2array(self, row):
        """Convert a row of cvs to a trajecotry point.
        
        Args:
            row (list): a row of csv file.

        Returns:
            np.array: point: delta_time(ms), delta_lng, delta_lat, sog, cog, time(ms), lng, lat
        """        
        array = np.array([row[6], row[7], row[8], row[2], row[5], row[3], row[4]], np.float32)
        return array

    def loadTestTrajectory(self, file_name):
        """Load trajectory data for testing. 

        trajectory.shape = [N, 7]
        The trajectories are converted to np.array([N, 7]), normalized and stored in self.trajectory. 
        (including lng, lat)

        Args:
            file_name (string): file name of the csv.
        """        
        self.file_name = file_name
        points_list = []
        with open(self.file_name, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                point = self.row2array(row)
                points_list.append(point)
        
        self.trajectory = np.array(points_list)

        # normalization: (x - min) / (max - min)
        self.max_train_data, self.min_train_data = self.trajectory.max(axis = 0), self.trajectory.min(axis = 0)
        self.trajectory = (self.trajectory - self.min_train_data) / (self.max_train_data - self.min_train_data)


    def getTestBP(self, batch_size, bp_step):
        """Get a batch of trajectory in bp_step and the point to predict for BP model testing.

        Args:
            batch_size (int): the size of mini-batch.
            bp_step (int): the length of trajectory sequence.

        Returns:
            x_test: sequence for testing. [Δtime, Δlng, Δlat, sog, cog]
            x_coordinates: coordinates of seq. [lng, lat]
        """           
        seq = []
        
        for i in range(batch_size):
            seq_temp = []
            is_valid = False
            while not is_valid:
                index = np.random.randint(0, len(self.trajectory) - bp_step)
                seq_temp = self.trajectory[index: index + bp_step]
                is_valid = True
                for point in seq_temp:
                    if point[0] == 0.0:
                        is_valid = False
                        break
            seq.append(seq_temp)
            x_test = np.array(seq)[:, :, :5]
            x_coordinates = np.array(seq)[:, :, 5:]
        return x_test, x_coordinates


    def getTestLSTM(self, batch_size, seq_length):
        """Get a batch of trajectory in seq_length and the point to predict for LSTM model testing.

        Args:
            batch_size (int): the size of mini-batch.
            seq_length (int): the length of trajectory sequence.

        Returns:
            x_test: sequence for testing. [Δtime, Δlng, Δlat, sog, cog]
            x_coordinates: coordinates of seq. [lng, lat]
        """        
        seq = []
        
        for i in range(batch_size):
            seq_temp = []
            is_valid = False
            while not is_valid:
                index = np.random.randint(0, len(self.trajectory) - seq_length)
                seq_temp = self.trajectory[index: index + seq_length]
                is_valid = True
                for point in seq_temp:
                    if point[0] == 0.0:
                        is_valid = False
                        break
            seq.append(seq_temp)
            x_test = np.array(seq)[:, :, :5]
            x_coordinates = np.array(seq)[:, :, 5:]
        return x_test, x_coordinates
    

    def getTestSeq2Seq(self, batch_size, encoder_length, decoder_length):
        """Get a batch of trajectory in encoder_length + decoder_length to predict for seq2seq model testing.

        Args:
            batch_size (int): the size of mini-batch.
            encoder_length (int): the length of source trajectory sequence for encoder.
            decoder_length (int): the length of destination trajectory sequence for decoder.

        Returns:
            seq_encoder_test: encoder sequence for testing. [Δtime, Δlng, Δlat, sog, cog].
            seq_encoder_coordinates: coordinates of encoder seq. [lng, lat].
            seq_decoder_test: decoder sequence for testing. [Δtime, Δlng, Δlat, sog, cog].
            seq_decoder_coordinates: coordinates of decoder seq. [lng, lat].
        """        
        seq_encoder = []
        seq_decoder = []
        seq_length = encoder_length + decoder_length
        
        for i in range(batch_size):
            seq_temp = []
            is_valid = False
            while not is_valid:
                index = np.random.randint(0, len(self.trajectory) - seq_length + 1)
                seq_temp = self.trajectory[index: index + seq_length]
                is_valid = True
                for point in seq_temp:
                    if point[0] == 0.0:
                        is_valid = False
                        break
            seq_encoder.append(seq_temp[:encoder_length, :])
            # 注意，这里seq_decoder的长度实际是decoder_length+1，因为seq_encoder的最后一位要用于输入
            seq_decoder.append(seq_temp[encoder_length-1:seq_length, :])
            seq_encoder_test = np.array(seq_encoder)[:, :, :5]
            seq_encoder_coordinates = np.array(seq_encoder)[:, :, 5:]
            seq_decoder_test = np.array(seq_decoder)[:, :, :5]
            seq_decoder_coordinates = np.array(seq_decoder)[:, :, 5:]
        return seq_encoder_test, seq_encoder_coordinates, seq_decoder_test, seq_decoder_coordinates

