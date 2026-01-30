import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

def data_provider(root_path, datasets, batch_size, win_size=100, step=100, flag="train", percentage=1):
    if flag == "train": shuffle = True
    else: shuffle = False
    print(f"loading {datasets}({flag}) percentage: {percentage*100}% ...", end="")
    data_path = os.path.join(root_path, datasets)
    border, border_1, border_2 = read_meta(datasets)
    data_set = UCRAnomalyloader(data_path, border, border_1, border_2, win_size, step, flag, percentage)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    print("done!")
    return data_set, data_loader

class UCRAnomalyloader(Dataset):
    def __init__(self, data_path, train_length, border_1, border_2, win_size, step, flag="train", percentage=0.1):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        # 1.read data
        data = read_data(data_path)
        label = np.zeros(data.shape[0])
        label[border_1:border_2] = 1
        # 2.train
        train_data = data[:train_length]
        train_label = label[:train_length]
        # 3.test
        test_data = data[train_length:]
        test_label = label[train_length:]
        # 4.process        
        self.scaler = StandardScaler()
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)

        if flag == "init":
            self.init = train_data
            self.init_label = train_label
        else:
            train_end = int(len(train_data) * 0.8)
            train_start = int(train_end*(1-percentage))
            self.train = train_data[train_start:train_end]
            self.train_label = train_label[train_start:train_end]
            self.val = train_data[train_end:]
            self.val_label = train_label[train_end:]
            self.test = test_data
            self.test_label = test_label

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "init":
            return (self.init.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":           
            return np.float32(self.train[index: index + self.win_size]), np.float32(self.train_label[index: index + self.win_size])
        elif self.flag == "val":
            return np.float32(self.val[index: index + self.win_size]), np.float32(self.val_label[index: index + self.win_size])
        elif self.flag == "test":
            return np.float32(self.test[index: index + self.win_size]), np.float32(self.test_label[index: index + self.win_size])
        elif self.flag == "init":
            return np.float32(self.init[index: index + self.win_size]), np.float32(self.init_label[index: index + self.win_size])
        else:
            return np.float32(self.test[index // self.step * self.win_size: index// self.step * self.win_size+ self.win_size]), np.float32(self.test_label[index // self.step * self.win_size: index // self.step * self.win_size + self.win_size])
    
def read_data(dataset_file_path):
    data_list = []
    try:
        with open(dataset_file_path, "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                data_list.append(data_line)
        data = np.stack(data_list, 0)
    except ValueError:
        with open(dataset_file_path, "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line[0].split()])
        data = data_line
        data = np.expand_dims(data, axis=1)
    return data

def read_meta(dataset):
    assert dataset.endswith('.txt')
    parts = dataset.split('_')
    if len(parts) < 3:
        return None

    border_str = parts[-3]
    border_1_str = parts[-2]
    border_2_str = parts[-1]
    if '.' in border_2_str:
        border_2_str = border_2_str[:border_2_str.find('.')]

    try:
        border = int(border_str)
        border_1 = int(border_1_str)
        border_2 = int(border_2_str)
        return border, border_1, border_2
    except ValueError:
        return None


