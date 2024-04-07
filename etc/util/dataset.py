import os
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader

from etc.util.datamanager import DataManager
import etc.util.hyper_parameters as hp
from etc.util.tokenize import tokenize


class ImdbDataset(Dataset):
    def __init__(self, train=True):
        self.data_path = hp.train_data_path if train else hp.test_data_path
        self.temp_data_path = [
            os.path.join(self.data_path, "pos"),
            os.path.join(self.data_path, "neg"),
        ]
        self.total_file_path = []
        for path in self.temp_data_path:
            self.file_name_list = os.listdir(path)
            self.file_path_list = [
                os.path.join(path, file_name)
                for file_name in self.file_name_list
                if file_name.endswith("txt")
            ]
            self.total_file_path.extend(self.file_path_list)

    def __getitem__(self, item):
        file_path = self.total_file_path[item]
        label_str = file_path.split("/")[-2]
        # neg: 0, pos: 1
        label = 0 if label_str == "neg" else 1
        token_list = tokenize(open(file_path, encoding="UTF-8").read())
        datamanager = pickle.load(open("./etc/util/imdb_dm.pkl", "rb"))
        vector = datamanager.token_vectorize(token_list, max_len=hp.max_len)
        return np.array(vector), np.array(label)

    def __len__(self):
        return len(self.total_file_path)

    def dataloader(self):
        data_loader = DataLoader(self, batch_size=hp.batch_size, shuffle=True)
        return data_loader


if __name__ == "__main__":
    dataset = ImdbDataset(True)
    print(dataset.__getitem__(0))
