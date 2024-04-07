# util cfg
train_data_path = "./etc/data/train"
test_data_path = "./etc/data/test"

# LSTM hyper parameters
max_len = 200  # 对齐每个句子的长度
batch_size = 128
max_features = 10000  # 词表的长度
embedding_size = 100  # embeeding的大小

lstm_hidden_size = 128  # LSTM 的单元数
lstm_num_layer = 2  # LSTM 的层数
lstm_bidirectional = True
lstm_dropout = 0.3

lstm_out_hidden = 10  # 输出层LSTM的单元数
lstm_out_num = 1  # 输出层LSTM的层数
lstm_out_bidirectional = False
lstm_out_dropout = 0.3

fc_dropout = 0.3

epoch = 100
output_class = 2
lr = 1e-3

# other model hyper parameters...