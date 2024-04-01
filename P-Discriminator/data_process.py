import random

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


def get_next_covariate(covariate_df: pd.DataFrame, horizon_size: int, quantile_size: int):

    # 将未来的视距内的数据关联到每个时间戳
    next_covariate = []
    covariate_size = covariate_df.shape[1]
    for i in range(1, covariate_df.shape[0] - horizon_size + 1):
        cur_covariate = []
        cur_covariate.append(covariate_df[i:i + horizon_size, :])
        next_covariate.append(cur_covariate)

    next_covariate = np.array(next_covariate)
    next_covariate = next_covariate.reshape(-1, horizon_size * covariate_size)

    return next_covariate


def split_data(data: pd.DataFrame, timestep, input_dim, horizon_size):
    x_data = []
    y_data = []

    for index in range(len(data) - timestep):
        # 存储窗口内的自变量数据
        x_data.append(data.iloc[index: index + timestep, 1:])
        # 存储未来的因变量数据
        y_data.append(data.iloc[index + timestep, 0])

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # 将未来的视距内的数据关联到每个时间戳

    next_x_data = []
    covariate_size = x_data.shape[2]
    for i in range(1, x_data.shape[0] - horizon_size + 1):
        cur_covariate = []
        cur_covariate.append(x_data[i:i + horizon_size, :])
        next_x_data.append(cur_covariate)

    next_x_data = np.array(next_x_data)
    next_x_data = next_x_data.reshape(-1, horizon_size * covariate_size)

    real_vals_list = []
    for i in range(1, horizon_size + 1):
        real_vals_list.append(np.array(y_data[i: y_data.shape[0] - horizon_size + i]))

    real_x_data = np.array(real_vals_list)
    real_x_data = real_x_data.reshape(-1, horizon_size)

    total_num = next_x_data.shape[0]
    total_num = total_num if total_num % 2 == 0 else total_num - 1

    # 获取训练集大小
    train_size = int(np.round(0.8 * total_num))
    train_size = train_size if train_size % 2 == 0 else train_size - 1

    # 划分训练集、测试集
    x_train = x_data[: train_size, :].reshape(-1, timestep, input_dim)
    y_train = y_data[: train_size]
    next_x_train = next_x_data[: train_size, :]
    real_x_train = real_x_data[: train_size, :]

    x_test = x_data[train_size: total_num, :].reshape(-1, timestep, input_dim)
    y_test = y_data[train_size: total_num]
    next_x_test = next_x_data[train_size: total_num, :]
    real_x_test = real_x_data[train_size: total_num, :]

    return x_train, y_train, x_test, y_test, next_x_train, next_x_test, real_x_train, real_x_test


def read_df(config: dict):
    """
        读取生成的测试数据
    """
    # 生成测试数据的时间索引范围
    time_range = pd.date_range('2010-01-01', '2020-12-01', freq='12h')
    # 总时间戳数
    time_len = len(time_range)
    # 生成目标变量时间序列Dataframe
    series_dict = {}
    for i in range(5):
        cur_vals = [np.sin(i*t) for t in range(time_len)]
        series_dict[i] = cur_vals
    target_df = pd.DataFrame(index=time_range, data=series_dict)

    # 生成外部变量时间序列Dataframe
    covariate_df = pd.DataFrame(
        index=target_df.index,
        data={'hour': target_df.index.hour,
              'dayofweek': target_df.index.dayofweek,
              'month': target_df.index.month}
    )

    # 对外部变量时间序列Dataframe进行标准化
    for col in covariate_df.columns:
        covariate_df[col] = (covariate_df[col] - np.mean(covariate_df[col])) / np.std(covariate_df[col])

    # # NEW 向后平移修正时间以便于用已知时间的数据预测未来的数据
    # shift_time = config['shift_time']
    # # 向后平移目标变量时间序列
    # target_df = target_df.shift(-shift_time).dropna()
    # # 删除外部变量时间序列最新的一段序列
    # covariate_df = covariate_df.iloc[:-shift_time, :]

    timestep = 1

    # 分割训练集和测试集
    horizon_size = config['horizon_size']

    train_target_df = target_df.iloc[:-horizon_size, :]
    test_target_df = target_df.iloc[-horizon_size:, :]
    train_covariate_df = covariate_df.iloc[:-horizon_size, :]
    test_covariate_df = covariate_df.iloc[-horizon_size:, :]

    # 截取100个样本
    truncation_size = 100
    small_train_target_df = train_target_df.iloc[-truncation_size:, :].copy()
    small_train_covariate_df = train_covariate_df.iloc[-truncation_size:, :].copy()

    return small_train_target_df, test_target_df, small_train_covariate_df, test_covariate_df


class MQRNN_dataset(Dataset):
    """
        MQRNN模型数据库的数据封装类
    """

    def __init__(self, target_df: pd.DataFrame, covariate_df: pd.DataFrame, horizon_size: int, quantile_size: int):
        self.target_df = target_df
        self.covariate_df = covariate_df
        self.horizon_size = horizon_size
        self.quantile_size = quantile_size

        # 将未来的视距内的数据关联到每个时间戳

        next_covariate = []
        covariate_size = self.covariate_df.shape[1]

        for i in range(1, self.covariate_df.shape[0] - horizon_size + 1):
            cur_covariate = []
            # for j in range(horizon_size):
            cur_covariate.append(self.covariate_df.iloc[i:i + horizon_size, :].to_numpy())
            next_covariate.append(cur_covariate)

        next_covariate = np.array(next_covariate)
        next_covariate = next_covariate.reshape(-1, horizon_size * covariate_size)

        self.next_covariate = next_covariate

    def __len__(self):
        # 目标变量的维度
        return self.target_df.shape[1]

    def __getitem__(self, idx):
        cur_series = np.array(self.target_df.iloc[:-self.horizon_size, idx])
        cur_covariate = np.array(self.covariate_df.iloc[:-self.horizon_size, :])  # covariate used in generating hidden states

        covariate_size = self.covariate_df.shape[1]
        # next_covariate = np.array(self.covariate_df.iloc[1:-self.horizon_size+1,:]) # covariate used in the MLP decoders

        real_vals_list = []
        for i in range(1, self.horizon_size + 1):
            real_vals_list.append(np.array(self.target_df.iloc[i: self.target_df.shape[0] - self.horizon_size + i, idx]))

        real_vals_array = np.array(real_vals_list)  # [horizon_size, seq_len]
        real_vals_array = real_vals_array.T  # [seq_len, horizon_size]
        cur_series_tensor = torch.tensor(cur_series)

        cur_series_tensor = torch.unsqueeze(cur_series_tensor, dim=1)  # [seq_len, 1]
        cur_covariate_tensor = torch.tensor(cur_covariate)  # [seq_len, covariate_size]
        cur_series_covariate_tensor = torch.cat([cur_series_tensor, cur_covariate_tensor], dim=1)
        next_covariate_tensor = torch.tensor(self.next_covariate)  # [seq_len, horizon_size * covariate_size]
        cur_real_vals_tensor = torch.tensor(real_vals_array)

        return cur_series_covariate_tensor, next_covariate_tensor, cur_real_vals_tensor
