import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
        编码器-解码器结构的编码器。对于MQRNN，该编码器与传统的seq2seq模型相同，后者基于LSTM
    """

    def __init__(self,
                 horizon_size: int,
                 covariate_size: int,
                 hidden_size: int,
                 dropout: float,
                 layer_size: int,
                 by_direction: bool,
                 device):

        super(Encoder, self).__init__()

        self.horizon_size = horizon_size
        self.covariate_size = covariate_size
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.by_direction = by_direction
        self.dropout = dropout
        self.LSTM = nn.LSTM(input_size=covariate_size,
                            hidden_size=hidden_size,
                            num_layers=layer_size,
                            dropout=dropout,
                            bidirectional=by_direction)

        # 设置模型初始权重概率分布
        for param in self.LSTM.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)

    def forward(self, x):
        seq_len = x.shape[0]
        batch_size = x.shape[1]
        direction_size = 2 if self.by_direction else 1

        y, _ = self.LSTM(x)

        y = y.view(seq_len, batch_size, direction_size, self.hidden_size)
        y = y[:, :, -1, :]
        y = y.view(seq_len, batch_size, self.hidden_size)

        return y


class GlobalDecoder(nn.Module):
    """
        基于编码器生成的隐藏张量和预测范围内的外部变量时间序列值
    """

    def __init__(self,
                 hidden_size: int,
                 covariate_size: int,
                 horizon_size: int,
                 context_size: int):

        super(GlobalDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.covariate_size = covariate_size
        self.horizon_size = horizon_size
        self.context_size = context_size

        self.linear1 = nn.Linear(in_features=hidden_size + covariate_size * horizon_size,
                                 out_features=horizon_size * hidden_size * 3)

        self.linear2 = nn.Linear(in_features=horizon_size * hidden_size * 3,
                                 out_features=horizon_size * hidden_size * 2)

        self.linear3 = nn.Linear(in_features=horizon_size * hidden_size * 2,
                                 out_features=(horizon_size + 1) * context_size)

        self.activation = nn.ReLU()

    def forward(self, input):
        """
            input shape: [seq_len, batch_size, hidden_size + covariate_size * horizon_size]
            output shape: [seq_len, batch_size, (horizon_size+1) * context_size]
        """

        layer1_output = self.linear1(input)
        layer1_output = self.activation(layer1_output)

        layer2_output = self.linear2(layer1_output)
        layer2_output = self.activation(layer2_output)

        layer3_output = self.linear3(layer2_output)
        layer3_output = self.activation(layer3_output)

        return layer3_output


class LocalDecoder(nn.Module):
    """
        基于全局解码器生成的张量和预测步骤中的外部变量时间序列值
    """

    def __init__(self, covariate_size, quantile_size, context_size, quantiles, horizon_size):

        super(LocalDecoder, self).__init__()
        self.covariate_size = covariate_size
        self.quantiles = quantiles
        self.quantile_size = quantile_size
        self.horizon_size = horizon_size
        self.context_size = context_size

        self.linear1 = nn.Linear(in_features=horizon_size * context_size + horizon_size * covariate_size + context_size,
                                 out_features=horizon_size * context_size)

        self.linear2 = nn.Linear(in_features=horizon_size * context_size,
                                 out_features=horizon_size * quantile_size)

        self.activation = nn.ReLU()

    def forward(self, input):
        """
            input_size: (horizon_size+1)*context_size + horizon_size*covariate_size
            output_size: horizon_size * quantile_size
        """

        layer1_output = self.linear1(input)
        layer1_output = self.activation(layer1_output)

        layer2_output = self.linear2(layer1_output)
        layer2_output = self.activation(layer2_output)

        return layer2_output


class QuantileLoss(nn.Module):
    def __init__(self, model):
        super(QuantileLoss, self).__init__()
        self.device = model.device
        self.quantile_size = model.quantile_size
        self.ldecoder = model.ldecoder

    def forward(self, x, y):
        y = torch.unsqueeze(y, dim=0)

        total_loss = torch.tensor([0.0], device=self.device)
        for i in range(self.quantile_size):
            p = self.ldecoder.quantiles[i]
            errors = y - x[:, :, :, i]
            cur_loss = torch.max((p - 1) * errors, p * errors)
            total_loss += torch.sum(cur_loss)

        return total_loss


class MQRNN(nn.Module):
    def __init__(
            self,
            horizon_size: int,
            hidden_size: int,
            quantiles: list,
            columns: list,
            dropout: float,
            layer_size: int,
            by_direction: bool,
            lr: float,
            batch_size: int,
            num_epochs: int,
            context_size: int,
            covariate_size: int,
            device
    ):
        super(MQRNN, self).__init__()
        self.device = device
        self.horizon_size = horizon_size
        self.quantile_size = len(quantiles)
        self.quantiles = quantiles
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.covariate_size = covariate_size
        quantile_size = self.quantile_size

        self.encoder = Encoder(horizon_size=horizon_size,
                               covariate_size=covariate_size,
                               hidden_size=hidden_size,
                               dropout=dropout,
                               layer_size=layer_size,
                               by_direction=by_direction,
                               device=device).to(device).double()

        self.gdecoder = GlobalDecoder(hidden_size=hidden_size,
                                      covariate_size=covariate_size,
                                      horizon_size=horizon_size,
                                      context_size=context_size).to(device).double()

        self.ldecoder = LocalDecoder(covariate_size=covariate_size,
                                     quantile_size=quantile_size,
                                     context_size=context_size,
                                     quantiles=quantiles,
                                     horizon_size=horizon_size).to(device).double()

    def forward(self, x, next_x):
        h = self.encoder(x)

        h = torch.unsqueeze(h[-1], dim=0)
        next_x = torch.unsqueeze(next_x, dim=0)

        h = torch.cat([h, next_x], dim=2)
        h = self.gdecoder(h)

        h = torch.cat([h, next_x], dim=2)
        h = self.ldecoder(h)

        y = h.view(h.shape[0], self.batch_size, self.horizon_size, self.quantile_size)

        return y

