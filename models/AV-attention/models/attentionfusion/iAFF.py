import torch
import torch.nn as nn


# 直接加法融合（DAF）
# 功能：将输入 x 和残差 residual 直接相加，实现最简单的特征融合。
class DAF(nn.Module):
    '''
    DirectAddFuse 直接加法融合
    '''

    def __init__(self):
        super(DAF, self).__init__()

    def forward(self, x, residual):
        return x + residual


# 注意力特征融合iAFF
# 功能：实现多次局部和全局注意力融合，首先通过局部和全局注意力机制生成权重，然后进行两次融合操作。
class iAFF(nn.Module):
    '''
    iAFF  multiple feature fusion
    '''
    # channels：输入特征图的通道数，默认为64
    # r：缩减比率，用于计算中间通道数 inter_channels
    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # first local attention
        # 定义了第一个局部注意力模块 self.local_att，由一系列层组成：
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),  # 1x1卷积，将通道数从 channels 降到 inter_channels。
            nn.BatchNorm2d(inter_channels),  # 批量归一化层
            nn.ReLU(inplace=True),  # ReLU 激活函数。
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),  # 1x1卷积，将通道数从 inter_channels 提升回 channels。
            nn.BatchNorm2d(channels),  # 批量归一化层。
        )

        # first global attention
        # 定义了第一个全局注意力模块 self.global_att，由一系列层组成：
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 自适应平均池化层，将特征图缩小到 1x1。
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),  # 1x1卷积，将通道数从 channels 降到 inter_channels。
            nn.BatchNorm2d(inter_channels),  # 批量归一化层。
            nn.ReLU(inplace=True),  # ReLU 激活函数。
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),  # 1x1卷积，将通道数从 inter_channels 提升回 channels。
            nn.BatchNorm2d(channels),  # 批量归一化层。
        )

        # second local attention
        # 定义了第二个局部注意力模块 self.local_att2，结构与第一个局部注意力模块相同。
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # second global attention
        # 定义了第二个全局注意力模块 self.global_att2，结构与第一个全局注意力模块相同。
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()  # 定义了一个 Sigmoid 激活函数，用于计算权重。

    # 在前向传播方法 forward 中，首先将输入 x 和残差 residual 相加得到 xa。然后通过局部注意力模块 self.local_att 和全局注意力模块 self.global_att 处理 xa，
    # 得到局部特征 xl 和全局特征 xg。将它们相加得到融合特征 xlg，通过 Sigmoid 激活函数计算权重 wei。最后，将输入 x 和残差 residual 按权重进行融合，得到 xi。
    # 接着，将中间结果 xi 通过第二个局部注意力模块 self.local_att2 和全局注意力模块 self.global_att2 处理，得到第二次融合特征 xlg2，
    # 通过 Sigmoid 激活函数计算第二次权重 wei2。最后，将输入 x 和残差 residual 按新的权重进行融合，得到最终输出 xo。
    def forward(self, x, residual):
        # print("Before fusion:")
        # print("x size:", x.size())  # ([64, 64, 2, 256])
        # print("residual size:", residual.size())  # ([64, 1, 2, 256])
        # if x.size(1) != residual.size(1):  # 通道数不匹配时调整
        #     residual = nn.Conv2d(residual.size(1), x.size(1), kernel_size=1, stride=1, bias=False).to(x.device)(residual)
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att2(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo
# 这个类通过局部和全局注意力机制对输入特征和残差特征进行多次融合。
# 注意力机制由一系列卷积层、批量归一化层和激活函数组成。
# 前向传播方法中，通过两次局部和全局注意力计算权重，并按权重融合输入和残差特征，得到最终输出。


# 注意力特征融合AFF
# 功能：与 iAFF 类似，但只进行一次局部和全局注意力融合。
class AFF(nn.Module):
    '''
    AFF  multiple feature fusion
    '''
    # 构造函数 __init__ 接受两个参数：channels：输入特征图的通道数，默认为64； r：缩减比率，用于计算中间通道数 inter_channels。
    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        # 局部注意力模块 self.local_att，由一系列层组成：
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),  # 1x1卷积，将通道数从 channels 降到 inter_channels。
            nn.BatchNorm2d(inter_channels),  # 批量归一化层。
            nn.ReLU(inplace=True),  # ReLU 激活函数。
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),  # 1x1卷积，将通道数从 inter_channels 提升回 channels。
            nn.BatchNorm2d(channels),  # 批量归一化层。
        )

        # 定义了全局注意力模块 self.global_att，由一系列层组成：
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 自适应平均池化层，将特征图缩小到 1x1。
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),  # 1x1卷积，将通道数从 channels 降到 inter_channels。
            nn.BatchNorm2d(inter_channels),  # 批量归一化层。
            nn.ReLU(inplace=True),  # ReLU 激活函数。
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),  # 1x1卷积，将通道数从 inter_channels 提升回 channels。
            nn.BatchNorm2d(channels),  # 批量归一化层。
        )

        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活函数，用于计算权重。

    # 在前向传播方法 forward 中，首先将输入 x 和残差 residual 相加得到 xa。
    # 然后通过局部注意力模块 self.local_att 和全局注意力模块 self.global_att 处理 xa，得到局部特征 xl 和全局特征 xg。
    # 将它们相加得到融合特征 xlg，通过 Sigmoid 激活函数计算权重 wei。
    # 接着，通过计算 2 * x * wei + 2 * residual * (1 - wei) 对输入 x 和残差 residual 按权重进行融合，得到最终输出 xo。
    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo