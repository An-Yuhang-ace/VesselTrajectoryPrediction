# Vesssel Trajectory Prediction

## 1， 数据预处理

### 训练轨迹数据载入类  

TrajectoryLoader.py
船舶轨迹数据载入类，包括用于训练的轨迹数据载入方法，用于各模型训练的getBatch方法。  

* row2array(row):行读取方法  
将读入的csv文件的一行解析为船舶轨迹点数据，数据类型为 np.float32
训练用数据结构为五维数据：Δtime, Δlng, Δlat, sog, cog  

* loadTrajectoryData(file_name)  
将file_name的csv文件中的船舶轨迹数据载入进轨迹数据类的trajectory变量中。
船舶轨迹数据点结构为：Δtime, Δlng, Δlat, sog, cog。
随后进行离差标准化归一化处理，normalization: (x - min)/(max - min)
用于归一化的数据保存在self.max_train_data and self.min_train_data  

* getBatchBP(batch_size, bp_step)
随机取一批用于BP模型训练的轨迹与真实点数据。
轨迹长度为bp_step。取出的轨迹需要保证是连续的。
Returns:
seq: shape of [batch_size, bp_step, 5].
next_point: shape of [batch_size, 5].  

* getBatchLSTM(batch_size, seq_length)
随机取一批用于LSTM模型训练的轨迹与真实点数据。
轨迹长度为seq_length。取出的轨迹需要保证是连续的。
Returns:
seq: shape of [batch_size, seq_length, 5].  
next_point: shape of [batch_size, 5].  

* getBatchSeq2Seq(batch_size, encoder_lenght, decoder_lenght)
随机取一批用于LSTM模型训练的源轨迹与目标序列数据。
轨迹长度为encoder_length和decoder_length。总轨迹需要是连续的。
（这里seq_decoder的长度实际是decoder_length+1，因为seq_encoder的最后一位要用于输入）
Returns:
seq_encoder: shape of [batch_size, encoder_length, 5].
seq_decoder: shape of [batch_size, decoder_length+1, 5].  

### 测试轨迹数据载入类

TestLoader.py
船舶轨迹数据载入类，包括用于测试的轨迹数据载入方法，用于各模型测试的getBatch方法。  

* row2array(row):行读取方法
将读入的csv文件的一行解析为船舶轨迹点数据，数据类型为 np.float32
测试用数据包括绝对坐标，结构为七维数据：Δtime, Δlng, Δlat, sog, cog, lng, lat.  

* getTestBP(batch_size, bp_step):
随机取一批用于BP模型测试的轨迹与真实点数据。
轨迹长度为bp_step。取出的轨迹需要保证是连续的。
返回为测试序列与序列绝对坐标。
Returns:
x_test: sequence for testing. [Δtime, Δlng, Δlat, sog, cog]
x_coordinates: coordinates of seq. [lng, lat]

* getTestLSTM(batch_size, seq_length):
随机取一批用于LSTM模型测试的轨迹与真实点数据。
轨迹长度为seq_length。取出的轨迹需要保证是连续的。
返回为测试序列与序列绝对坐标。
Returns:
x_test: sequence for testing. [Δtime, Δlng, Δlat, sog, cog]
x_coordinates: coordinates of seq. [lng, lat]

* getTestSeq2Seq(batch_size, encoder_lenght, decoder_lenght):
随机取一批用于LSTM模型测试的源轨迹与目标序列数据。
轨迹长度为encoder_length和decoder_length。总轨迹需要是连续的。
返回为编码序列，编码序列绝对坐标，解码序列，解码序列绝对坐标。
Returns:
seq_encoder_test: encoder sequence for testing. [Δtime, Δlng, Δlat, sog, cog].
seq_encoder_coordinates: coordinates of encoder seq. [lng, lat].
seq_decoder_test: decoder sequence for testing. [Δtime, Δlng, Δlat, sog, cog].
seq_decoder_coordinates: coordinates of decoder seq. [lng, lat].  

## 2, 网路模型

### LSTM

Model: "lstm"
_________________________________________________________________
Layer (type)                 Output Shape              Param #

lstm_1 (LSTM)                multiple                  81408

output_layer (Dense)         multiple                  645
_________________________________________________________________
Total params: 82,053
Trainable params: 82,053
Non-trainable params: 0
_________________________________________________________________

超参数：batch_size, lstm_step, n_lstm
lstm层：units = n_lstm, dropout=0.1
output层：units = 5

call:
初始状态为零张量
直接输入lstm层，再输入输出层，返回即可。

### Encoder

Model: "encoder"
_________________________________________________________________
Layer (type)                 Output Shape              Param #

lstm (LSTM)                  multiple                  68608
_________________________________________________________________

Total params: 68,608
Trainable params: 68,608
Non-trainable params: 0
_________________________________________________________________

超参数：batch_size, n_lstm
lstm层：units = n_lstm, dropout=0.1, return_sequences=True, return_state=True
output层：units = 5

call:
初始状态为零张量
直接输入lstm层，返回lstm_out and states 即可。

### Decoder

Model: "decoder"
_________________________________________________________________
Layer (type)                 Output Shape              Param #

lstm_1 (LSTM)                multiple                  68608

output_layer (Dense)         multiple                  645
_________________________________________________________________
Total params: 69,253
Trainable params: 69,253
Non-trainable params: 0
_________________________________________________________________

超参数：batch_size, n_lstm
lstm层：units = n_lstm, dropout=0.1, return_sequences=True, return_state=True
output层：units = 5

call:
初始状态来自输入
直接输入lstm层，返回lstm_out and states，lstm_out通过输出层
返回输出和状态。

### Attention

### DecoderAttention

Model: "decoder_attention"   n_lstm = 128, attention_func = 'general'
_________________________________________________________________
Layer (type)                 Output Shape              Param #

attention (Attention)        multiple                  16512

lstm_1 (LSTM)                multiple                  68608

wc_layer (Dense)             multiple                  32896

output_layer (Dense)         multiple                  645
_________________________________________________________________
Total params: 118,661
Trainable params: 118,661
Non-trainable params: 0
_________________________________________________________________

若 attention_func = 'dot', attention层无参数
若 attention_func = 'concat', attention层33025个参数

超参数：batch_size, n_lstm, attention_func
lstm层：units = n_lstm, dropout=0.1, return_sequences=True, return_state=True
output层：units = 5

call:
初始状态来自输入
注意力层输入：lstmout和encoder_output
直接输入lstm层，返回lstm_out and states，lstm_out通过输出层
返回输出和状态。

## 3，网络训练

### BPtrain.py

### LSTMtrain.py

1. 设置模型，训练参数， 优化方法：
n_lstm, seq_length, lstm_step
learning_rate, batch_size, num_batches
optimizer一般选择Adam

2. 建立并初始化模型
建立模型：lstm neural_net.
通过输入来预先制定输入形状
随后可对网络进行summary.

3. 设置tensorboard & checkpoint
tensorboard用于可视化训练过程
checkpoint保存训练后的参数，便于测试时重载。

4. 定义训练过程
step处理方法：将 (batch_size, seq_length, 5)的航迹数据处理按步输入的形式
 (batch_size, seq_length - lstm_step+1, lstm_step*5)
LSTM训练过程：
loss为解码器输出序列和真实序列的MSE（这里只选择Δlng, Δlat二维数据）
对编解码器参数进行优化

5. 载入训练集进行训练
实例化航迹数据载入类
用StepProcess函数将航迹数据重构为每step步一个输入
进行训练，每display_step个batch记录一个checkpoint。

### Seq2Seqtrain.py

1. 设置模型，训练参数， 优化方法：
n_lstm, encoder_length, decoder_length
learning_rate, batch_size, num_batches
optimizer一般选择Adam
注意，此处发现需要更多batch_num来对模型进行训练，目前估计大约需要10w

2. 建立并初始化模型
建立模型：encoder and decoder.
通过输入来预先制定输入形状
随后可对网络进行summary.

3. 设置tensorboard & checkpoint
tensorboard用于可视化训练过程
checkpoint保存训练后的参数，便于测试时重载。

4. 定义Seq2Seq训练过程
decoder段需要一个循环来实现时序预测，loss累加最后取平均。
loss为解码器输出序列和真实序列的RMSE（这里只选择 Δlng, Δlat二维数据）
递归预测时将真实值用于预测（Scheduled Sampling有效性待验证）
对编解码器参数进行优化

5. 载入训练集进行训练
实例化航迹数据载入类
进行训练，每display_step个batch记录一个checkpoint。

### AttentionSeq2Seqtrain.py

1. 设置模型，训练参数， 优化方法：
n_lstm, encoder_length, decoder_length
learning_rate, batch_size, num_batches
optimizer一般选择Adam

2. 建立并初始化模型
建立模型：encoder_a and decoder_a.
通过输入来预先制定输入形状
随后可对网络进行summary.

3. 设置tensorboard & checkpoint
tensorboard用于可视化训练过程
checkpoint保存训练后的参数，便于测试时重载。

4. 定义Attention-Seq2Seq训练过程
decoder段需要一个循环来实现时序预测，loss累加最后取平均。
loss为解码器输出序列和真实序列的MSE（这里只选择Δt, Δlng, Δlat三维数据）
递归预测时将真实值用于预测（Scheduled Sampling有效性待验证）
对编解码器参数进行优化

5. 载入训练集进行训练
实例化航迹数据载入类
进行训练，每display_step个batch记录一个checkpoint。

## 4，模型测试

### TestPoints.py

多点预测测试，测试模型的实际预测能力

1. 创建模型，重载参数
重载模型:lstm, encoder, decoder, encoder_a, decoder_a

2. 定义预测方法

    * TestSeq2Seq
    进行Seq2Seq轨迹预测，返回平均RMSE loss。

    * TestSeq2SeqAttention
    进行AttentionSeq2Seq轨迹预测，返回平均RMSE loss。  

    * TestLSTM
    进行LSTM轨迹预测，用时序循环实现多点预测。
    返回平均RMSE loss。

3. 载入测试数据，设定测试参数
载入测试轨迹集，设置进行预测的源序列长度和目标序列长度

4. 进行预测
随机获取batch_size的连续轨迹序列，包括用于预测的源序列，用于验证的目标序列
用于输出绝对坐标。
分布用LSTM, Seq2Seq, AttentionSeq2Seq进行预测，loss为平均RMSE。

### TestVisual.py

预测点可视化测试

1. 创建模型，重载参数
重载模型:lstm, encoder, decoder, encoder_a, decoder_a

2. 定义预测方法

    * TestSeq2Seq
    进行Seq2Seq轨迹预测，用时序循环进行预测，保存预测点
    返回预测点和平均loss
    Returns:
    pred [np.array(pred)]: The prediction of points. shape of [seq_length, 5].
    loss [tensor]: Root Mean Squre Error loss of prediction of points.

    * TestSeq2SeqAttention
    进行AttentionSeq2Seq轨迹预测，用时序循环进行预测，保存预测点
    返回预测点和平均loss
    Returns:
    pred [np.array(pred)]: The prediction of points. shape of [seq_length, 5].
    loss [tensor]: Root Mean Squre Error loss of prediction of points.

    * TestLSTM
    进行LSTM轨迹预测，用时序循环进行预测，保存预测点
    返回预测点和平均loss
    Returns:
    pred [np.array(pred)]: The prediction of points. shape of [seq_length, 5].
    loss [tensor]: Root Mean Squre Error loss of prediction of points.

3. 载入测试数据，设定测试参数
载入测试轨迹集，设置进行预测的源序列长度和目标序列长度

4. 进行预测
随机获取batch_size的连续轨迹序列，包括用于预测的源序列，用于验证的目标序列
用于输出绝对坐标。
进行预测，pred为预测结果，loss为平均MSE。

5. 绝对坐标恢复与可视化
源坐标：载入，去归一，转为list，用simplekml可视化
真实坐标：载入，去归一，转为list，用simplekml可视化
预测坐标：载入，去归一，转为list，用simplekml可视化
