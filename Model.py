import tensorflow as tf
import numpy as np
import random

class BP(tf.keras.Model):
    """3-layer BP network model for trajectory prediction.
    """    
    def __init__(self, n_lstm, step, batch_size):        
        super().__init__()
        self.batch_size = batch_size
        self.step = step
        self.fc = tf.keras.layers.Dense(n_lstm, activation = tf.nn.relu, name="hidden_layer")
        self.out = tf.keras.layers.Dense(5, name="output_layer")

    @tf.function
    def call(self, x):
        # x.shape: (batch_size, step, 5), y.shape: (batch_size, step*5)
        y = tf.reshape(x, [self.batch_size, self.step * 5])
        z = self.fc(y)
        # z.shape: (batch_size, n_lstm) pre.shape:(batch_size, 5)
        pred = self.out(z)
        return pred

'''
class LSTMFixed(tf.keras.Model):
    """3-layer LSTM-RNN model for trajectory prediction.(The length of input is fixed)

    Build hidden layer with keras.layers.LSTMCell.
    """    
    def __init__(self, n_hidden, seq_length, step, batch_size):
        super().__init__()
        self.step = step
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.lstm = tf.keras.layers.LSTMCell(units = n_hidden, activation = tf.nn.relu)
        self.out = tf.keras.layers.Dense(units = 5, name="output_layer")

    @tf.function
    def call(self, x):
        state = self.lstm.get_initial_state(batch_size = self.batch_size, dtype = tf.float32)
        for t in range(self.seq_length - self.step + 1):
            # take "step" points as input of lstm, y.shape = [batch_size, step, 5] z.shape = [batch_size, step * 5]
            y = x[:, t : t+self.step, :]
            z = tf.reshape(y, [self.batch_size, self.step * 5])
            output, state = self.lstm(z, state)
            pred = self.out(output)
        return pred 
'''


class LSTM(tf.keras.Model):
    """3-layer LSTM-RNN model for trajectory prediction.
    (The input length is not fixed, but the lstm_step is fixed to 1)

    Build hidden layer with keras.layers.LSTM.
    """    
    def __init__(self, n_lstm, step, batch_size):
        super().__init__()
        self.step = step
        self.batch_size = batch_size
        self.lstm_size = n_lstm
        self.lstm = tf.keras.layers.LSTM(units = n_lstm, activation = tf.nn.relu, dropout=0.1)
        self.out = tf.keras.layers.Dense(units = 5, name="output_layer")

    @tf.function
    def call(self, x):
        # x.shape: (batch_size, seq_length-step+1, step*5), y.shape: (batchsize, n_lstm)
        state = self.init_states(self.batch_size)
        y = self.lstm(x, initial_state = state)
        # pred.shape: (batch_size, 5)
        pred = self.out(y)
        return pred 

    def init_states(self, batch_size):
        return (tf.zeros([batch_size, self.lstm_size]), tf.zeros([batch_size, self.lstm_size]))

class Encoder(tf.keras.Model):
    """Encoder of seq2seq model for trajectory prediction.
    Build RNN layer with keras.layers.LSTM.
    """    
    def __init__(self, n_lstm, batch_size):
        super().__init__()
        self.n_lstm = n_lstm
        self.batch_size = batch_size
        self.lstm = tf.keras.layers.LSTM(self.n_lstm, return_sequences=True, return_state=True, activation=tf.nn.relu, dropout=0.1)

    @tf.function
    def call(self, sequence):
        states = self.init_states()
        # sequence.shape: (batch_size, encoder_length, 5), output.shape: (batch_size, encoder_length, n_lstm)
        # state_h, state_c has shape of (batch_size, n_lstm)
        output, state_h, state_c = self.lstm(sequence, initial_state=states)
        return output, state_h, state_c

    @tf.function
    def init_states(self):
        return(tf.zeros([self.batch_size, self.n_lstm]), tf.zeros([self.batch_size, self.n_lstm]))

'''
class Decoder_LSTMCell(tf.keras.Model):
    """Decoder of seq2seq model for trajectory prediction.
    Build hidden layer with keras.layers.LSTMCell. The input_seq is a full seq.
    Implement scheduled sampling when is_training is True. 
    """    
    def __init__(self, n_lstm, batch_size):
        super().__init__()
        self.n_lstm = n_lstm
        self.batch_size = batch_size
        self.lstm = tf.keras.layers.LSTMCell(n_lstm, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units = 5)

    @tf.function
    def call(self, seq_in, seq_length, state, is_training=False):
        #state = self.lstm.get_initial_state(batch_size = self.batch_size, dtype = tf.float32)
        x_sample = np.zeros([self.batch_size, 5])
        seq_out = []
        for t in range(seq_length):
            if is_training:
                # x_true的第一位从seq_encoder最后一位开始。
                x_true = seq_in[:, t, :]
                # TODO scheduled sampling 每次训练时，投一枚硬币，决定采用sample值还是true值。
                # 如果可以的话，随着训练的进行，我们希望这个值为sample值的概率逐渐增大。
                if random.randint(0, 1) > 0.5 or t == 0:
                    x = x_true
                else:
                    x = x_sample
            else:
                # predict
                x = x_sample
            y, state = self.lstm(x, state)
            pred = self.out(y)
            x_sample = pred
            seq_out.append(tf.reshape(pred, [self.batch_size, 1, 5]))
        return tf.concat(seq_out, 1)
'''

class Decoder(tf.keras.Model):
    """Decoder of seq2seq model for trajectory prediction.
    Build RNN layer with keras.layers.LSTM.
    The decoder can only process one point, so you need call it in a loop when traning and predicting.
    """    
    def __init__(self, n_lstm, batch_size):
        super().__init__()
        self.n_lstm = n_lstm
        self.batch_size = batch_size
        self.lstm = tf.keras.layers.LSTM(self.n_lstm, return_sequences=True, return_state=True, activation=tf.nn.relu, dropout=0.1)
        self.out = tf.keras.layers.Dense(units = 5, name="output_layer")

    def call(self, seq_in, state):
        # seq_in.shape: (batch_size, 1, 5), lstm_out.shape：(batch_size, 1, n_lstm)
        lstm_out, state_h, state_c = self.lstm(seq_in, initial_state=state)
        logits = self.out(lstm_out)
        # logits.shape: (batch_size, 1, 5), reshape it to (batch_size, 5)
        logits = tf.reshape(logits, [self.batch_size, 5])
        return logits, state_h, state_c


class Attention(tf.keras.Model):
    """Attention layer of seq2seq model for trajectory prediction.
    Choose between three score function: ['dot', 'general', 'concat']
    The number of parameters of three function: [0, n_lstm*(n_lstm+1), 2*(n_lstm * (n_lstm+1))+1]
    """      
    def __init__(self, n_lstm, attention_func):
        super().__init__()
        self.attention_func = attention_func
        if attention_func not in ['dot', 'general', 'concat']:
            raise ValueError(
                'Unknown attention score function! Must be either dot, general or concat.')
        if attention_func == 'general':
            # General score function
            self.wa = tf.keras.layers.Dense(n_lstm)
        elif attention_func == 'concat':
            # Concat score function
            self.wa = tf.keras.layers.Dense(n_lstm, activation='tanh')
            self.va = tf.keras.layers.Dense(1)
    
    def call(self, decoder_output, encoder_output):
        if self.attention_func == 'dot':
            # dot score function: decoder_output (dot) encoder_output
            # decoder_output.shape: (batch_size, 1, n_lstm), encoder_output.shape: (batch_size, encoder_length, n_lstm)
            # => score.shape: (batch_size, 1, encoder_length)
            score = tf.matmul(decoder_output, encoder_output, transpose_b=True)
        elif self.attention_func == 'general':
            # general score function: decoder_output (dot) (Wa (dot) encoder_output)
            # decoder_output.shape: (batch_size, 1, n_lstm), encoder_output.shape: (batch_size, encoder_length, n_lstm)
            # => score.shape: (batch_size, 1, encoder_length)
            score = tf.matmul(decoder_output, self.wa(encoder_output), transpose_b=True)
        elif self.attention_func == 'concat':
            # concat score function: va (dot) tanh(Wa (dot) concat(decoder_out + encoder_output))
            # decoder_output must be broadcasted to encoder output's shape first
            decoder_output = tf.tile(decoder_output, [1, encoder_output.shape[1], 1])
            # tf.concat => Wa => va
            # (batch_size, encoder_output, n_lstm * 2) => (batch_size, encoder_output, n_lstm) => (batch_size, encoder_output, 1)
            score = self.va(self.wa(tf.concat((decoder_output, encoder_output), axis=-1)))
            # transose to the shape of (batch_size, 1, encoder_length)
            score = tf.transpose(score, [0, 2, 1])
        
        alignment = tf.nn.softmax(score, axis=2)
        # context vector c_t is the weighted average sum of encoder output
        context = tf.matmul(alignment, encoder_output)
        return context, alignment


class DecoderAttention(tf.keras.Model):
    """Decoder with Attention mechanism of seq2seq model for trajectory prediction.
    The decoder can only process one point, so you need call it in a loop when traning and predicting.
    Build RNN layer with keras.layers.LSTM.
    """    
    def __init__(self, n_lstm, batch_size, attention_func):
        super().__init__()
        self.attention = Attention(n_lstm, attention_func)
        self.n_lstm = n_lstm
        self.batch_size = batch_size
        self.lstm = tf.keras.layers.LSTM(self.n_lstm, return_sequences=True, return_state=True, activation=tf.nn.relu, dropout=0.1)
        self.wc = tf.keras.layers.Dense(n_lstm, activation='tanh', name="wc_layer")
        self.out = tf.keras.layers.Dense(units = 5, name="output_layer")

    def call(self, seq_in, state, encoder_output):
        # lstm_out.shape: (batch_size, 1, n_lstm)
        lstm_out, state_h, state_c = self.lstm(seq_in, initial_state=state)
        # context.shape: (batch_size, 1, n_lstm)
        # alignment.shape: (batch_size, 1, seq_length)
        context, alignment = self.attention(lstm_out, encoder_output)
        # Combine the context vector and the LSTM output
        lstm_out = tf.concat([tf.squeeze(context, 1), tf.squeeze(lstm_out, 1)], 1)
        # lstm_out now has shape (batch_size, rnn_size)
        lstm_out = self.wc(lstm_out)
        # Finally, it is converted back to trajectory space: (batch_size, 5)
        logits = self.out(lstm_out)

        return logits, lstm_out, state_h, state_c, alignment

    

