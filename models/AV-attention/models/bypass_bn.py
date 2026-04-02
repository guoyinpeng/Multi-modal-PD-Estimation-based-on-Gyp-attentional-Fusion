
# NOTE：定义了两个函数，分别用于禁用和启用神经网络模型中 BatchNorm2d 层的运行统计信息。
#  这两个函数会遍历模型中的所有层，并在需要时修改 BatchNorm2d 层的 momentum 属性。

import torch
import torch.nn as nn


# 禁用模型中 BatchNorm2d 层的运行统计信息
def disable_running_stats(model):
    # 检查每个子模块是否是 nn.BatchNorm2d 类型。如果是则备份它的 momentum 属性，然后将 momentum 设置为 0，从而禁用该层的运行统计信息更新。
    def _disable(module):
        if isinstance(module, nn.BatchNorm2d):
            module.backup_momentum = module.momentum  # 备份原始 momentum
            module.momentum = 0  # 将 momentum 设置为 0

    model.apply(_disable)  # 将 _disable 函数应用于模型的每一层
# model.apply 方法会递归地遍历模型的所有子模块，并对每个子模块应用 _disable 函数。


# 恢复模型中 BatchNorm2d 层的运行统计信息。
def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, nn.BatchNorm2d) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum  # 恢复原始 momentum

    model.apply(_enable)  # 将 _enable 函数应用于模型的每一层
