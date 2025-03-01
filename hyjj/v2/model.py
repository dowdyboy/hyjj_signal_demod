# Description: This file contains the model architecture for the BiLSTM model with attention mechanism.
# Date: 2024-9-22
# Author: Wenhao Liu

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out


class CNNFeatureExtractor(nn.Module):

    def __init__(self, hidden_chanel=32, num_layer=2, drop_rate=0.3):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=hidden_chanel, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_chanel)

        ch = hidden_chanel
        self.layers = nn.ModuleList()
        for _ in range(num_layer):
            self.layers.append(
                ResidualBlock(ch, ch * 2)
            )
            ch = ch * 2

        # self.res_block1 = ResidualBlock(32, 64)
        # self.res_block2 = ResidualBlock(64, 128)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Conv1d(in_channels=ch, out_channels=ch, kernel_size=1, padding=0, stride=1)
        )

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.bn1(self.conv1(x))))

        for layer in self.layers:
            x = layer(x)
            x = self.pool(x)

        # x = self.res_block1(x)
        # x = self.pool(x)
        #
        # x = self.res_block2(x)
        # x = self.pool(x)

        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        weights = torch.softmax(self.attn(lstm_output), dim=1)
        weighted_output = torch.sum(weights * lstm_output, dim=1)
        return weighted_output


class BiLSTMModel(nn.Module):
    def __init__(self, feature_extractor, lstm_input_size, hidden_size, num_layers=2, num_classes=10):
        super(BiLSTMModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.lstm_input_size = lstm_input_size
        self.lstm = nn.LSTM(input_size=self.lstm_input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=True,
                            batch_first=True)
        self.attention = Attention(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.attention(x)
        x = self.fc(x)
        return x


class BiLSTMModel2(nn.Module):
    def __init__(self, feature_extractor, lstm_input_size, hidden_size, num_layers=2, num_classes=10):
        super(BiLSTMModel2, self).__init__()
        self.feature_extractor = feature_extractor
        self.lstm_input_size = lstm_input_size

        self.lstm_amr = nn.LSTM(input_size=self.lstm_input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=True,
                            batch_first=True)
        self.attention_amr = Attention(hidden_size * 2)
        self.fc_amr = nn.Linear(hidden_size * 2, num_classes)

        self.lstm_cw = nn.LSTM(input_size=self.lstm_input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                bidirectional=True,
                                batch_first=True)
        self.attention_cw = Attention(hidden_size * 2)
        self.fc_cw = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.permute(0, 2, 1)

        x_amr, _ = self.lstm_amr(x)
        x_amr = self.attention_amr(x_amr)
        x_amr = self.fc_amr(x_amr)

        x_cw, _ = self.lstm_cw(x)
        x_cw = self.attention_cw(x_cw)
        x_cw = self.fc_cw(x_cw)
        x_cw = torch.clip(x_cw, min=0., max=1.)
        x_cw = torch.squeeze(x_cw, dim=1)

        return x_amr, x_cw

