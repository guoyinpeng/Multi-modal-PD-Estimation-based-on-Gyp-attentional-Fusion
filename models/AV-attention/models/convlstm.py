import math
import torch
import torch.nn as nn


def init_layer(layer):
    """He init for Conv/Linear."""
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
    elif layer.weight.ndimension() == 3:
        (n_out, n_in, height) = layer.weight.size()
        n = n_in * height
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2.0 / n)
    scale = std * math.sqrt(3.0)
    layer.weight.data.uniform_(-scale, scale)
    if layer.bias is not None:
        layer.bias.data.fill_(0.0)


def init_att_layer(layer):
    layer.weight.data.fill_(1.0)
    if layer.bias is not None:
        layer.bias.data.fill_(0.0)


def init_bn(bn):
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class ConvBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, pad, normalisation, dil=1):
        super(ConvBlock1d, self).__init__()
        self.norm = normalisation
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            dilation=dil,
        )
        if self.norm == "bn":
            self.bn1 = nn.BatchNorm1d(out_channels)
        elif self.norm == "wn":
            self.conv1 = nn.utils.weight_norm(self.conv1, name="weight")
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        if self.norm == "bn":
            init_bn(self.bn1)

    def forward(self, input):
        x = input
        if self.norm == "bn":
            x = self.relu(self.bn1(self.conv1(x)))
        else:
            x = self.relu(self.conv1(x))
        return x


class ConvBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, pad, normalisation, att=None):
        super(ConvBlock2d, self).__init__()
        self.norm = normalisation
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
        )
        if self.norm == "bn":
            self.bn1 = nn.BatchNorm2d(out_channels)
        elif self.norm == "wn":
            self.conv1 = nn.utils.weight_norm(self.conv1, name="weight")
        self.att = att
        if not self.att:
            self.act = nn.ReLU()
        else:
            self.norm = None
            if self.att == "softmax":
                self.act = nn.Softmax(dim=-1)
            elif self.att == "global":
                self.act = None
            else:
                self.act = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        if self.att:
            init_att_layer(self.conv1)
        else:
            init_layer(self.conv1)
        if hasattr(self, "bn1") and self.bn1 is not None:
            init_bn(self.bn1)

    def forward(self, input):
        x = input
        if self.att:
            x = self.conv1(x)
            if self.act is not None:
                x = self.act(x)
        else:
            if self.norm == "bn":
                x = self.act(self.bn1(self.conv1(x)))
            else:
                x = self.act(self.conv1(x))
        return x


class FullyConnected(nn.Module):
    def __init__(self, in_channels, out_channels, activation, normalisation, att=None):
        super(FullyConnected, self).__init__()
        self.att = att
        self.norm = normalisation
        self.fc = nn.Linear(in_features=in_channels, out_features=out_channels)
        if activation == "sigmoid":
            self.act = nn.Sigmoid()
            self.norm = None
        elif activation == "softmax":
            self.act = nn.Softmax(dim=-1)
            self.norm = None
        elif activation == "global":
            self.act = None
            self.norm = None
        else:
            self.act = nn.ReLU()
            if self.norm == "bn":
                self.bnf = nn.BatchNorm1d(out_channels)
            elif self.norm == "wn":
                self.wnf = nn.utils.weight_norm(self.fc, name="weight")
        self.init_weights()

    def init_weights(self):
        if self.att:
            init_att_layer(self.fc)
        else:
            init_layer(self.fc)
        if getattr(self, "norm", None) == "bn" and hasattr(self, "bnf"):
            init_bn(self.bnf)

    def forward(self, input):
        x = input
        if self.norm is not None:
            if self.norm == "bn":
                x = self.act(self.bnf(self.fc(x)))
            else:
                x = self.act(self.wnf(x))
        else:
            if self.att:
                x = self.act(self.fc(x)) if self.act else self.fc(x)
            else:
                x = self.act(self.fc(x)) if self.act else self.fc(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        return x + self.pos_embedding[:, : x.size(1), :]


class EnhancedTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        num_layers,
        dim_feedforward=2048,
        dropout=0.1,
        max_len=500,
        learnable_pe=True,
    ):
        super().__init__()
        self.pos_encoder = (
            LearnablePositionalEncoding(d_model, max_len)
            if learnable_pe
            else PositionalEncoding(d_model, max_len)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.pos_encoder(x)
        return self.transformer(x)


class ConvLSTM_Visual(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        conv_hidden,
        transformer_dim,
        num_layers,
        activation,
        norm,
        dropout,
        pool_type="mean",
        dim_feedforward=None,
        learnable_pe=True,
    ):
        super(ConvLSTM_Visual, self).__init__()
        ffn = dim_feedforward if dim_feedforward is not None else transformer_dim
        self.conv = ConvBlock2d(
            in_channels=input_dim,
            out_channels=conv_hidden,
            kernel=(87, 3),
            stride=(1, 1),
            pad=(0, 1),
            normalisation="bn",
        )
        self.pool = nn.MaxPool1d(kernel_size=3, stride=3, padding=0)
        self.drop = nn.Dropout(dropout)
        self.transformer = EnhancedTransformerEncoder(
            d_model=conv_hidden,
            nhead=4,
            num_layers=num_layers,
            dim_feedforward=ffn,
            dropout=dropout,
            max_len=500,
            learnable_pe=learnable_pe,
        )
        self.fc = FullyConnected(
            in_channels=conv_hidden,
            out_channels=output_dim,
            activation=activation,
            normalisation=norm,
        )
        self.pool_type = pool_type
        if pool_type == "attention":
            self.attention_pool = nn.Linear(conv_hidden, 1)

    def forward(self, net_input, return_sequence=False):
        x = net_input
        batch, C, F, T = x.shape
        x = self.conv(x)
        x = self.pool(x.squeeze())
        x = self.drop(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.transformer(x)
        if return_sequence:
            return x
        if self.pool_type == "last":
            x = x[:, -1, :]
        elif self.pool_type == "mean":
            x = x.mean(dim=1)
        elif self.pool_type == "attention":
            attn_weights = torch.softmax(self.attention_pool(x), dim=1)
            x = (x * attn_weights).sum(dim=1)
        x = self.fc(x)
        return x


class ConvLSTM_Audio(nn.Module):
    def __init__(self, input_dim, output_dim, conv_hidden, lstm_hidden, num_layers, activation, norm, dropout):
        super(ConvLSTM_Audio, self).__init__()
        self.conv = ConvBlock1d(
            in_channels=input_dim,
            out_channels=conv_hidden,
            kernel=3,
            stride=1,
            pad=1,
            normalisation="bn",
        )
        self.pool = nn.MaxPool1d(kernel_size=3, stride=3, padding=0)
        self.drop = nn.Dropout(dropout)
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=conv_hidden,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=True,
        )
        self.fc = FullyConnected(
            in_channels=lstm_hidden * 2,
            out_channels=output_dim,
            activation=activation,
            normalisation=norm,
        )

    def forward(self, net_input, return_sequence=False):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv(x)
        x = self.pool(x)
        x = self.drop(x)
        x = x.permute(0, 2, 1).contiguous()
        x, _ = self.lstm(x)
        if return_sequence:
            return x
        x = self.fc(x[:, -1, :].reshape(batch, -1))
        return x
