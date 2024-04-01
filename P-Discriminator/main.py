import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from tqdm import tqdm
import logging

from config import CONFIG
from data_process import *
from MQRNN import MQRNN, QuantileLoss


# 日志信息常量类
class LogInfo:
    # 日志级别
    level = logging.DEBUG
    # 日志输出格式
    format = "%(asctime)s - %(levelname)s - %(message)s"


# 配置日志基本信息
logging.basicConfig(
    filename="../log/app.log",
    level=LogInfo.level,
    format=LogInfo.format
)

# 配置日志处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(LogInfo.level)
console_handler.setFormatter(logging.Formatter(LogInfo.format))

# 初始化日志记录仪
logger = logging.getLogger()
logger.addHandler(console_handler)


# 主程序
if __name__ == '__main__':
    # 获取模型配置常量数据

    horizon_size = CONFIG['horizon_size']
    quantiles = CONFIG['quantiles']
    quantile_size = len(quantiles)
    hidden_size = CONFIG['hidden_size']
    columns = CONFIG['columns']
    dropout = CONFIG['dropout']
    layer_size = CONFIG['layer_size']
    by_direction = CONFIG['by_direction']
    lr = CONFIG['lr']
    batch_size = CONFIG['batch_size']
    num_epochs = CONFIG['num_epochs']
    context_size = CONFIG['context_size']
    timestep = CONFIG['timestep']

    save_path = "./{}.pth".format("MQRNN")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = "../data/wind_dataset.csv"

    # 读取原始数据
    df = pd.read_csv(data_path, index_col=0).dropna()
    logger.info("读取原始数据成功: {}".format(data_path))
    logger.info("dataframe表头: {}".format(df.columns.values))
    logger.info("dataframe示例数据: \n{}".format(df.head()))

    covariate_size = 4
    input_dim = 4

    # dataframe -> array
    # 分割数据集为训练集、测试集
    x_train_array, y_train_array, x_test_array, y_test_array, next_x_train_array, next_x_test_array, real_x_train_array, real_x_test_array = split_data(df, timestep, input_dim, horizon_size)
    logger.info("分割数据集为训练集、测试集")
    logger.info("shape of x_train_array: {}".format(x_train_array.shape))
    logger.info("shape of y_train_array: {}".format(y_train_array.shape))
    logger.info("shape of x_test_array: {}".format(x_test_array.shape))
    logger.info("shape of y_test_array: {}".format(y_test_array.shape))
    logger.info("shape of next_x_train_array: {}".format(next_x_train_array.shape))
    logger.info("shape of next_x_test_array: {}".format(next_x_test_array.shape))
    logger.info("shape of real_x_train_array: {}".format(real_x_train_array.shape))
    logger.info("shape of real_x_test_array: {}".format(real_x_test_array.shape))

    # array -> tensor
    # 强制修改数据集的数据格式为double
    x_train_tensor = torch.from_numpy(x_train_array).to(torch.double)
    x_test_tensor = torch.from_numpy(x_test_array).to(torch.double)
    y_train_tensor = torch.from_numpy(y_train_array).to(torch.double)
    y_test_tensor = torch.from_numpy(y_test_array).to(torch.double)
    next_x_train_tensor = torch.from_numpy(next_x_train_array).to(torch.double)
    next_x_test_tensor = torch.from_numpy(next_x_test_array).to(torch.double)
    real_x_train_array = torch.from_numpy(real_x_train_array).to(torch.double)
    real_x_test_array = torch.from_numpy(real_x_test_array).to(torch.double)

    # tensor -> TensorDataset
    # 封装每个时间窗口内的数据为一份
    train_data = TensorDataset(x_train_tensor, y_train_tensor, next_x_train_tensor, real_x_train_array)
    test_data = TensorDataset(x_test_tensor, y_test_tensor, next_x_test_tensor, real_x_test_array)

    # TensorDataset -> DataLoader
    # 根据batch重新封装
    train_loader = torch.utils.data.DataLoader(train_data, batch_size, False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size, False)

    # 定义模型
    model = MQRNN(horizon_size, hidden_size, quantiles, columns, dropout, layer_size, by_direction, lr, batch_size,
                  num_epochs, context_size, covariate_size, device).to(device)
    # 定义损失函数
    loss_function = QuantileLoss(model).to(device)
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = 1
    for epoch in range(num_epochs):
        # 模型训练
        model.train()
        running_loss = 0
        test_bar = tqdm(train_loader)
        for data in test_bar:
            x_train, y_train, next_x_train, real_y_train = data
            x_train = x_train.permute(1, 0, 2).to(device)
            next_x_train = next_x_train.to(device)
            y_train = y_train.to(device)
            real_y_train = real_y_train.to(device)
            optimizer.zero_grad()
            y_train_pred = model(x_train, next_x_train)
            loss = loss_function(y_train_pred, real_y_train)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            test_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, num_epochs, loss.item())

        # 模型测试
        model.eval()
        test_loss = 0
        with torch.no_grad():
            test_bar = tqdm(test_loader)
            for data in test_bar:
                x_test, y_test, next_x_test, real_y_test = data
                x_test = x_test.permute(1, 0, 2).to(device)
                next_x_test = next_x_test.to(device)
                y_test = y_test.to(device)
                real_y_test = real_y_test.to(device)
                optimizer.zero_grad()
                y_test_pred = model(x_test, next_x_test)
                loss = loss_function(y_test_pred, real_y_test)

                test_loss += loss.item()
                test_bar.desc = "test epoch[{}/{}] loss:{:.3f}".format(epoch + 1, num_epochs, loss.item())

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), save_path)

    print('Finished Training')





