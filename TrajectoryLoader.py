import tensorflow as tf
import numpy as np
import csv
import time

class TrajectoryLoader():
    def __init__(self):
        super().__init__()

    def row2array(self, row):
        """Convert a row of csv to a trajecotry point.
        
        Args:
            row (list): a row of csv file.

        Returns:
            np.array: point: delta_time(ms), delta_lng, delta_lat, sog, cog
        """        
        array = np.array([row[6], row[7], row[8], row[2], row[5]], np.float32)
        return array


    def loadTrajectoryData(self, file_name):
        """Load trajectory data for LSTM/BP network. 

        trajectory.shape = [N, 5]
        The trajectories are converted to np.array([N, 5]), normalized and stored in self.trajectory. 
        (including break points, whose point[0] == 0.0)

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

    
    def getBatchBP(self, batch_size, bp_step):
        """Get a batch of trajectory in bp_step and the point to predict for BP model training.

        Args:
            batch_size (int): the size of mini-batch.
            bp_step (int): the length of trajectory sequence.

        Returns:
            seq: shape of [batch_size, bp_step, 5].
            next_point: shape of [batch_size, 5].
        """           
        seq = []
        next_point = []
        
        for i in range(batch_size):
            seq_temp = []
            # make sure the suquence is continuous.
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
            next_point.append(self.trajectory[index + bp_step]) 
        # array(seq).shape: [batch_size, bp_step, 5], array(next_point).shape: [batch_size, 5]
        return np.array(seq), np.array(next_point)


    def getBatchLSTM(self, batch_size, seq_length):
        """Get a batch of trajectory in seq_length and the point to predict for LSTM model training.

        Args:
            batch_size (int): the size of mini-batch.
            seq_length (int): the length of trajectory sequence.

        Returns:
            seq: shape of [batch_size, seq_length, 5]. 
            next_point: shape of [batch_size, 5].
        """        
        seq = []
        next_point = []
        
        for i in range(batch_size):
            seq_temp = []
            # make sure the suquence is continuous.
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
            next_point.append(self.trajectory[index + seq_length]) 
        # array(seq).shape: [batch_size, seq_length, 5], array(next_point).shape: [batch_size, 5]
        return np.array(seq), np.array(next_point)
    

    def getBatchSeq2Seq(self, batch_size, encoder_length, decoder_length):
        """Get a batch of trajectory in encoder_length + decoder_length to predict for seq2seq model training.

        Args:
            batch_size (int): the size of mini-batch.
            encoder_length (int): the length of source trajectory sequence for encoder.
            decoder_length (int): the length of destination trajectory sequence for decoder.

        Returns:
            seq_encoder: shape of [batch_size, encoder_length, 5].
            seq_decoder: shape of [batch_size, decoder_length+1, 5].
        """        
        seq_encoder = []
        seq_decoder = []
        seq_length = encoder_length + decoder_length
        
        for i in range(batch_size):
            seq_temp = []
            is_valid = False
            # make sure the suquence is continuous.
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
        # array(seq_encoder).shape: [batch_size, encoder_length, 5], array(seq_decoder).shape: [batch_size, decoder_length+1, 5]
        return np.array(seq_encoder), np.array(seq_decoder)


'''
    def loadDataBP(self, file_name, step):
        """Load training data for BP network.

        Take "step" points as x_train, 1 point as y_train.
        The data is converted to np.array, normalized, and stored in self.train_data.

        Args:
            file_name (string): file_name of the csv
            step (int): input steps.
        """        
        self.file_name = file_name
        self.step = step
        x_list = []
        y_list = [] 
        with open(self.file_name, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            points = []

            # take "step" points as x_train, 1 point as y_train
            for i in range(self.step + 1):
                point = self.row2array(rows[i])
                points.append(point)

            for i in range(self.step + 1, len(rows)):
                is_valid = True
                # skip the points including break point, which are not consecutive.
                for point in points:
                    if point[0] == 0.0:
                        is_valid = False
                        break
                if is_valid: 
                    # reshape the array and append it to the list
                    x = np.array(points[0: self.step])
                    # x.shape = [30, ]
                    x = x.reshape([5 * self.step, ])
                    x_list.append(x)
                    # y.shape = [5, ]
                    y = points[-1]
                    y_list.append(y)

                for j in range(self.step):
                    points[j] = points[j+1]
                points[self.step] = self.row2array(rows[i])

        self.x_train = np.array(x_list)
        self.y_train = np.array(y_list)
        self.size_train_data = self.x_train.shape[0]

        # normalization: (x - min) / (max - min)
        self.max_x_train, self.min_x_train = self.x_train.max(axis = 0), self.x_train.min(axis = 0)
        self.max_y_train, self.min_y_train = self.y_train.max(axis = 0), self.y_train.min(axis = 0)
        self.x_train = (self.x_train - self.min_x_train) / (self.max_x_train - self.min_x_train)
        self.y_train = (self.y_train - self.min_y_train) / (self.max_y_train - self.min_y_train)
        
        # x_train.shape = [N, 30], y_train.shape = [N, 5]
        self.train_data = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))  
'''


if __name__ == '__main__':
    x = TrajectoryLoader()
    start = time.clock()
    x.loadTrajectoryData("./DataSet/TrajectoryMillion.csv")
    end = time.clock()
    print("running time: %s s" % (end-start))
    seq = x.getBatchSeq2Seq(1024, 40, 20)

