import torch
import torch.nn as nn
import pandas as pd
import numpy as np


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
        self.LSTM = nn.LSTM(input_size=covariate_size + 1,
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

    def forward(self, input):
        """
            input shape: [seq_len, batch_size, covariate_size + 1]
            output shape: [seq_len, batch_size, hidden_size]
        """

        seq_len = input.shape[0]
        batch_size = input.shape[1]
        direction_size = 1
        if self.by_direction:
            direction_size = 2

        outputs, _ = self.LSTM(input)

        outputs_reshape = outputs.view(seq_len, batch_size, direction_size, self.hidden_size)

        outputs_last_layer = outputs_reshape[:, :, -1, :]

        final_outputs = outputs_last_layer.view(seq_len, batch_size, self.hidden_size)

        return final_outputs
