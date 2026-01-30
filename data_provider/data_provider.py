import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

warnings.filterwarnings('ignore')


def data_provider(root_path, datasets, batch_size, win_size=100, step=100, flag="train", percentage=1):
    if flag == "train":
        shuffle = True
    else:
        shuffle = False
    print(f"loading {datasets}({flag}) percentage: {percentage * 100}% ...", end="")
    file_paths, train_lens = read_meta(root_path=root_path, dataset=datasets)
    discrete_channels = None
    if datasets == "MSL": discrete_channels = range(1, 55)
    if datasets == "SMAP": discrete_channels = range(1, 25)
    if datasets == "SWAT": discrete_channels = [2, 4, 9, 10, 11, 13, 15, 19, 20, 21, 22, 29, 30, 31, 32, 33, 42, 43, 48,
                                                50]
    data_set = TrainSegLoader(file_paths, train_lens, win_size, step, flag, percentage, discrete_channels)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    print("done!")
    return data_set, data_loader


class TrainSegLoader(Dataset):
    def __init__(self, data_path, train_length, win_size, step, flag="train", percentage=0.1, discrete_channels=None):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        # 1.read data
        data = read_data(data_path)
        # 2.train
        train_data = data.iloc[:train_length, :]
        train_data, train_label = (
            train_data.loc[:, train_data.columns != "label"].to_numpy(),
            train_data.loc[:, ["label"]].to_numpy(),
        )
        # 3.test
        test_data = data.iloc[train_length:, :]
        test_data, test_label = (
            test_data.loc[:, test_data.columns != "label"].to_numpy(),
            test_data.loc[:, ["label"]].to_numpy(),
        )
        # 4.process
        if discrete_channels is not None:
            train_data = np.delete(train_data, discrete_channels, axis=-1)
            test_data = np.delete(test_data, discrete_channels, axis=-1)

        self.scaler = StandardScaler()
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)

        if flag == "init":
            self.init = train_data
            self.init_label = train_label
        else:
            train_end = int(len(train_data) * 0.8)
            train_start = int(train_end * (1 - percentage))
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

    def __getitem__(self, index, eps=1):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index: index + self.win_size]), np.float32(
                self.train_label[index: index + self.win_size])
        elif self.flag == "val":
            return np.float32(self.val[index: index + self.win_size]), np.float32(
                self.val_label[index: index + self.win_size])
        elif self.flag == "test":
            return np.float32(self.test[index: index + self.win_size]), np.float32(
                self.test_label[index: index + self.win_size])
        elif self.flag == "init":
            return np.float32(self.init[index: index + self.win_size]), np.float32(
                self.init_label[index: index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size: index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_label[index // self.step * self.win_size: index // self.step * self.win_size + self.win_size])


def read_data(path: str, nrows=None) -> pd.DataFrame:
    data = pd.read_csv(path)
    label_exists = "label" in data["cols"].values
    all_points = data.shape[0]
    columns = data.columns
    if columns[0] == "date":
        n_points = data.iloc[:, 2].value_counts().max()
    else:
        n_points = data.iloc[:, 1].value_counts().max()
    is_univariate = n_points == all_points
    n_cols = all_points // n_points
    df = pd.DataFrame()
    cols_name = data["cols"].unique()
    if columns[0] == "date" and not is_univariate:
        df["date"] = data.iloc[:n_points, 0]
        col_data = {
            cols_name[j]: data.iloc[j * n_points: (j + 1) * n_points, 1].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    elif columns[0] != "date" and not is_univariate:
        col_data = {
            cols_name[j]: data.iloc[j * n_points: (j + 1) * n_points, 0].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
    elif columns[0] == "date" and is_univariate:
        df["date"] = data.iloc[:, 0]
        df[cols_name[0]] = data.iloc[:, 1]
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    else:
        df[cols_name[0]] = data.iloc[:, 0]
    if label_exists:
        last_col_name = df.columns[-1]
        df.rename(columns={last_col_name: "label"}, inplace=True)
    if nrows is not None and isinstance(nrows, int) and df.shape[0] >= nrows:
        df = df.iloc[:nrows, :]
    return df


def read_meta(root_path, dataset):
    meta_path = root_path + "/DETECT_META.csv"
    meta = pd.read_csv(meta_path)
    meta = meta.query(f'file_name.str.contains("{dataset}")', engine="python")
    file_paths = root_path + f"/data/{meta.file_name.values[0]}"
    train_lens = meta.train_lens.values[0]
    return file_paths, train_lens
