"""
建立词表，实现字符与数字之间的相互转换并保存词表
利用 训练集 中的数据建立词表，跟测试集没有关系
"""

import os
import pickle
from tqdm import tqdm
from etc.util.tokenize_rw import tokenize
import etc.util.hyper_parameters as hp


class DataManager:
    def __init__(self):
        self.UNK_TAG = "UNK"
        self.PAD_TAG = "PAD"
        self.UNK = 0
        self.PAD = 1
        self.vocab = {self.UNK_TAG: self.UNK, self.PAD_TAG: self.PAD}
        self.count = {}
    
    def __len__(self):
        return len(self.vocab)

    def count_token_frequence(self, token_list):
        for token in token_list:
            self.count[token] = self.count.get(token, 0) + 1

    def build_vocab(self, min=0, max=None, max_features=None):
        if min is not None:
            self.count = {
                token: value for token, value in self.count.items() if value > min
            }

        if max is not None:
            self.count = {
                token: value for token, value in self.count.items() if value < max
            }

        if max_features is not None:
            temp = sorted(self.count.items(), key=lambda x: x[-1], reverse=True)[
                :max_features
            ]
            self.count = dict(temp)

        for token in self.count:
            self.vocab[token] = len(self.vocab)

        self.inverse_vocab = dict(zip(self.vocab.values(), self.vocab.keys()))

    def token_vectorize(self, token_list, max_len=None):
        # 统一长度
        if max_len is not None:
            if max_len > len(token_list):
                token_list += [self.PAD_TAG] * (max_len - len(token_list))
            if max_len < len(token_list):
                token_list = token_list[:max_len]

        return [self.vocab.get(token, self.UNK) for token in token_list]

    def token_devectorize(self, vector):
        return [self.inverse_vocab.get(idx) for idx in vector]


if __name__ == "__main__":
    datamanager = DataManager()
    path = hp.train_data_path
    temp_data_path = [
        os.path.join(hp.train_data_path, "pos"),
        os.path.join(hp.train_data_path, "neg"),
    ]
    for data_path in temp_data_path:
        file_paths = [
            os.path.join(data_path, file_name)
            for file_name in os.listdir(data_path)
            if file_name.endswith("txt")
        ]
        for file_path in tqdm(file_paths):
            token_list = tokenize(open(file_path, encoding="UTF-8").read())
            datamanager.count_token_frequence(token_list)
    datamanager.build_vocab(max_features=hp.max_features)
    pickle.dump(datamanager, open("./etc/util/imdb_dm.pkl", "wb"))
    print(len(datamanager))
