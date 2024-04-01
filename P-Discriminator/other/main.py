from data_process import MQRNN_dataset, read_df
import torch
import matplotlib.pyplot as plt

from config import CONFIG


def p_discriminator_train():
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

    # 获取目标变量的训练集、测试集和外部变量的训练集、测试集
    train_target_df, test_target_df, train_covariate_df, test_covariate_df = read_df(CONFIG)

    covariate_size = train_covariate_df.shape[1]

    # 获取硬件驱动
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 定义模型
    mqrnn_model = MQRNN(horizon_size, hidden_size, quantiles, columns, dropout, layer_size, by_direction, lr, batch_size,
                        num_epochs, context_size, covariate_size, device)

    # 定义数据类实例
    train_dataset = MQRNN_dataset(train_target_df, train_covariate_df, horizon_size, quantile_size)

    # 模型训练
    mqrnn_model.train(train_dataset)

    predict_result = mqrnn_model.predict(train_target_df, train_covariate_df, test_covariate_df, columns)

    # 模型预测结果可视化

    plt.rcParams.update({'figure.autolayout': True})
    plt.figure(figsize=(15, 5), dpi=300)
    plt.plot(predict_result[quantiles[0]], color='r', label='prediction')
    plt.plot(test_target_df[0].to_list(), color='b', label='real')
    plt.xticks([i for i in range(len(test_target_df))], [str(x) for x in test_target_df.index.values], rotation=-90)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    p_discriminator_train()
