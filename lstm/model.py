import torch.nn as nn
import torch.nn.functional as F

import etc.util.hyper_parameters as hp


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(
            hp.max_features, hp.embedding_size
        )
        self.lstm = nn.LSTM(
            input_size=hp.embedding_size,
            hidden_size=hp.lstm_hidden_size,
            num_layers=hp.lstm_num_layer,
            batch_first=True,
            bidirectional=hp.lstm_bidirectional,
            dropout=hp.lstm_dropout,
        )
        self.lstm_out = nn.LSTM(
            input_size=hp.lstm_hidden_size * 2,
            hidden_size=hp.lstm_out_hidden,
            num_layers=hp.lstm_out_num,
            batch_first=True,
            bidirectional=hp.lstm_out_bidirectional,
        )
        # dropout=hp.lstm_out_dropout 单词 RNN 中不允许添加dropout
        self.fc = nn.Sequential(
            nn.Linear(hp.lstm_out_hidden, hp.output_class),
            nn.Sigmoid(),
            nn.BatchNorm1d(hp.output_class),
            nn.Dropout(hp.fc_dropout),
        )

    def forward(self, input):
        # [batch_size, max_len] -> [batch_size, max_len, embedding_size]
        x = self.embedding(input)
        # out: [batch_size, max_len, hidden_size*2] h_n: [2*2, batch_size, hidden_size]
        x, (h_n, c_n) = self.lstm(x)
        # out: [batch_size, max_len, hidden_size]
        x_out, (h_out_n, c_out_n) = self.lstm_out(x)
        # 获取最后一次输出
        # h_n [1, batch_size, hidden_size]

        # [batch_size, hidden_size]
        output = h_out_n[0]
        out = self.fc(output)
        return F.log_softmax(out, dim=-1)
