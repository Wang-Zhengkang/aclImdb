train_data_path = "./etc/data/train"
test_data_path = "./etc/data/test"

max_len = 200  # 对齐每个句子的长度
batch_size = 128
max_features = 10000  # 词表的长度
embedding_size = 100  # embeeding 的大小

lstm_hidden_size = 128  # LSTM 的单元数
lstm_num_layer = 2  # LSTM 的层数
lstm_bidriectional = True  # LSTM 是否为双向
lstm_dropout = 0.3  # 随机 dropout

lstm_out_hidden = 10  # 输出层LSTM的单元数
lstm_out_num = 1  # 输出层LSTM的层数
lstm_out_bidriectional = False  # 输出层LSTM为单向
lstm_out_dropout = 0.3  # 输出层LSTM的dropout

fc_dropout = 0.3  # 全连接层的dropout

epoch = 100  # 训练迭代次数
output_class = 2  # 判别类别
lr = 1e-3
