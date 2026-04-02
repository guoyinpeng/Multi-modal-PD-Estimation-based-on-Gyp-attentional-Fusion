from itertools import count
import csv
import json
import os
import sys
import random
import logging
import numpy as np
from datetime import datetime
from scipy import stats
from sklearn import metrics

import torch
import torch.nn as nn
import torch_optimizer as optim_ext
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from torch.utils.data import WeightedRandomSampler, DataLoader, ConcatDataset, Subset
from torchvision import transforms

# local functions
from dataset.dataset import PDDataset, ToTensor
from models.convlstm import ConvLSTM_Visual, ConvLSTM_VisualBiLSTM
from models.evaluator import Evaluator
from models.SAM import SAM
from models.Lookhead import Lookahead


# 这个函数的目的是通过设置随机种子来确保代码在不同运行中的一致性和可重复性。
# 这在机器学习和深度学习实验中尤为重要，因为它可以确保实验结果的可复现性，从而更可靠地进行模型评估和调试。
def init_seed(manual_seed):
    """
    Set random seed for torch and numpy.
    """
    manual_seed = int(manual_seed)
    random.seed(manual_seed)  # 设置 Python 的内置随机数生成器的种子。
    np.random.seed(manual_seed)  # 设置 NumPy 的随机数生成器的种子。
    torch.manual_seed(manual_seed)  # 设置 PyTorch CPU 上随机数生成器的种子。

    torch.cuda.manual_seed_all(manual_seed)  # 设置 PyTorch 所有 GPU 设备上的随机数生成器的种子。
    torch.backends.cudnn.deterministic = True  # 设置 cuDNN 后端为确定性模式。
    torch.backends.cudnn.benchmark = False  # 禁用 cuDNN 的自动优化功能。


# 日志记录器
def get_logger(filepath, log_title):
    logger = logging.getLogger(filepath)  # 创建了一个日志记录器对象，该日志记录器的名称由参数 filepath 指定。
    logger.setLevel(logging.INFO)  # 设置了日志记录器的日志级别为 INFO，这意味着只有 INFO 级别及以上的日志信息才会被记录。
    fh = logging.FileHandler(filepath)  # 创建了一个文件处理器对象 用于将日志信息写入到指定的文件 filepath 中
    fh.setLevel(logging.INFO)  # 这一行设置了文件处理器的日志级别为 INFO，与之前设置的日志记录器保持一致
    logger.addHandler(fh)  # 将文件处理器 fh 添加到日志记录器 logger 中，以便将日志信息写入到文件中。
    logger.info('-' * 54 + log_title + '-' * 54)
    return logger


# 打印和记录日志信息
def log_and_print(logger, msg):
    logger.info(msg)  # 使用参数传入的日志记录器 logger，调用其 info 方法，记录一条日志信息。信息内容为参数 msg 所指定的内容。
    # print(msg)  # 打印参数 msg 所指定的内容到标准输出。


# 数据加载器的工作函数
def worker_init_fn(worker_id):
    """
    Init worker in dataloader.
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)
# 这个函数的作用是确保在使用数据加载器进行多进程数据加载时，每个工作进程的随机数生成器有不同的种子，从而生成不同的随机数序列。
# 这样可以避免数据加载过程中的随机性冲突，提高数据加载的多样性和可靠性。


# 加权随机采样器
def get_sampler_pd_binary(pd_binary_gt):
    # sampler for pd_binary_gt
    # 针对二分类标签的采样器
    class_sample_count = np.unique(pd_binary_gt, return_counts=True)[1]  # class_sample_count 保存了每个类别的样本数量
    weight = 1. / class_sample_count  # 计算每个类别的权重，权重为类别样本数量的倒数 [这样，样本数量少的类别将具有更高的权重，以便在采样时被更频繁地选中，平衡数据集]
    samples_weight = weight[pd_binary_gt]  # weight[pd_binary_gt] 通过索引将每个标签对应的权重分配给 samples_weight。
    samples_weight = torch.from_numpy(samples_weight).double()  # 将 samples_weight 转换为一个PyTorch张量，并设置数据类型为 double（浮点型）。

    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))  # 创建一个 WeightedRandomSampler 实例，用于在采样时考虑样本的权重。
    return sampler


def get_sampler_pd_score(pd_score_gt):
    # class_sample_ID保存了每个唯一标签的值；class_sample_count保存了每个标签的样本数量
    class_sample_ID, class_sample_count = np.unique(pd_score_gt, return_counts=True)
    weight = 1. / class_sample_count
    samples_weight = np.zeros(pd_score_gt.shape)  # 创建一个与 pd_score_gt 形状相同的零数组 samples_weight，用于存储每个样本的权重。
    for i, sample_id in enumerate(class_sample_ID):  # 使用 enumerate 对 class_sample_ID 进行遍历，i 是索引，sample_id 是类别标签值。
        indices = np.where(pd_score_gt == sample_id)[0]
        value = weight[i]
        samples_weight[indices] = value
    samples_weight = torch.from_numpy(samples_weight).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler
# 这个函数的作用是根据给定的 pd_score_gt 生成一个加权随机采样器，以平衡不同类别的样本在采样时的概率。这样可以在训练模型时平衡类别不平衡的问题。

def _to_scalar(x):
    if torch.is_tensor(x):
        if x.numel() == 1:
            return x.item()
        return x.reshape(-1)[0].item()
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return x.item()
        return x.reshape(-1)[0].item()
    return x

def _build_full_dataset(data_config):
    """
    构建用于五折交叉验证的完整数据集
    优先使用 ALL_ROOT_DIR；
    如果没有，则将原 train/test 两个根目录拼接起来。
    """
    transform = transforms.Compose([ToTensor()])

    all_root_dir = data_config.get('ALL_ROOT_DIR', None)
    if all_root_dir is not None and str(all_root_dir).strip() != '':
        full_dataset = PDDataset(
            all_root_dir,
            'train',
            visual_with_face3d=data_config['VISUAL_WITH_FACE3D'],
            transform=transform
        )
    else:
        train_dataset = PDDataset(
            data_config['TRAIN_ROOT_DIR'],
            'train',
            visual_with_face3d=data_config['VISUAL_WITH_FACE3D'],
            transform=transform
        )
        test_dataset = PDDataset(
            data_config['TEST_ROOT_DIR'],
            'test',
            visual_with_face3d=data_config['VISUAL_WITH_FACE3D'],
            transform=transform
        )
        full_dataset = ConcatDataset([train_dataset, test_dataset])

    return full_dataset


def _extract_ids_and_labels(dataset):
    """
    从 dataset 中提取每个样本对应的 subject ID 和 pd_binary_gt
    这里不依赖 PDDataset 的内部成员变量，只依赖 __getitem__()
    """
    ids = []
    labels = []

    for i in range(len(dataset)):
        sample = dataset[i]
        sample_id = int(_to_scalar(sample['ID']))
        sample_label = int(_to_scalar(sample['pd_binary_gt']))
        ids.append(sample_id)
        labels.append(sample_label)

    return np.array(ids), np.array(labels)

# 数据加载器的构建（五折交叉验证）
def get_dataloaders(data_config, fold_idx=0, n_splits=5, seed=42, val_ratio=0.2):
    """Build train / val / test loaders for one CV fold."""
    full_dataset = _build_full_dataset(data_config)
    ids, labels = _extract_ids_and_labels(full_dataset)

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )

    splits = list(splitter.split(np.zeros(len(labels)), labels, groups=ids))
    train_idx, test_idx = splits[fold_idx]

    train_ids_sub = ids[train_idx]
    train_labels_sub = labels[train_idx]
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    inner_train_rel, inner_val_rel = next(
        gss.split(np.zeros(len(train_labels_sub)), train_labels_sub, groups=train_ids_sub)
    )
    train_final_idx = train_idx[inner_train_rel]
    val_idx = train_idx[inner_val_rel]

    train_dataset = Subset(full_dataset, train_final_idx.tolist())
    val_dataset = Subset(full_dataset, val_idx.tolist())
    test_dataset = Subset(full_dataset, test_idx.tolist())

    train_labels = labels[train_final_idx]
    sampler = get_sampler_pd_binary(train_labels)

    bs = data_config['BATCH_SIZE']
    nw = data_config['NUM_WORKERS']
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=bs,
            num_workers=nw,
            sampler=sampler,
            worker_init_fn=worker_init_fn,
            pin_memory=torch.cuda.is_available()
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=bs,
            num_workers=nw,
            shuffle=False,
            worker_init_fn=worker_init_fn,
            pin_memory=torch.cuda.is_available()
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=bs,
            num_workers=nw,
            shuffle=False,
            worker_init_fn=worker_init_fn,
            pin_memory=torch.cuda.is_available()
        )
    }

    fold_info = {
        'fold_idx': fold_idx + 1,
        'train_size': len(train_final_idx),
        'val_size': len(val_idx),
        'test_size': len(test_idx),
        'train_subjects': len(np.unique(ids[train_final_idx])),
        'val_subjects': len(np.unique(ids[val_idx])),
        'test_subjects': len(np.unique(ids[test_idx])),
        'train_pos': int(np.sum(labels[train_final_idx] == 1)),
        'train_neg': int(np.sum(labels[train_final_idx] == 0)),
        'val_pos': int(np.sum(labels[val_idx] == 1)),
        'val_neg': int(np.sum(labels[val_idx] == 0)),
        'test_pos': int(np.sum(labels[test_idx] == 1)),
        'test_neg': int(np.sum(labels[test_idx] == 0)),
        'train_labels': labels[train_final_idx],
    }

    return dataloaders, fold_info

# 查找最新的检查点
def find_last_ckpts(path, key, date=None):
    # path：存储检查点文件的文件夹路径
    # key：模型的类型（如模型名称或其他标识符） 用于在文件夹中筛选相关的文件
    # date（可选）：指定的日期 用于筛选该日期相关的检查点文件
    """Finds the last checkpoint file of the last trained model in the
    model directory.
    Arguments:
        path: str, path to the checkpoint
        key: str, model type
        date: str, a specific date in string format 'YYYY-MM-DD'
    Returns:
        The path of the last checkpoint file
    """
    ckpts = list(sorted(os.listdir(path))) # 获取目录中所有文件列表并排序

    # 验证日期的格式是否正确
    if date is not None:
        # match the date format
        date_format = "%Y-%m-%d"
        try:
            datetime.strptime(date, date_format)
            # print("This is the correct date string format.")
            matched = True
        except ValueError:
            # print("This is the incorrect date string format. It should be YYYY-MM-DD")
            matched = False
        assert matched, "The given date is the incorrect date string format. It should be YYYY-MM-DD"

        # 根据日期调整 key
        key = '{}_{}'.format(key, date)
    else:
        key = str(key)

    # 根据 key 筛选文件
    ckpts = list(filter(lambda f: f.startswith(key), ckpts))
    # 获取最新的检查点
    last_ckpt = os.path.join(path, ckpts[-1]) #由于进行了排序，最新的检查点文件位于 ckpts 列表的最后一个位置

    return last_ckpt  # 返回最新的检查点路径


# 获取模型及其权重（仅视觉分支 DSFE + PDAF-Net 分类头）
def get_models(model_config, args, model_type=None, ckpt_path=None):
    vn = model_config['VISUAL_NET']
    bb = vn.get('BACKBONE', 'transformer').lower()
    if bb == 'transformer':
        visual_net = ConvLSTM_Visual(
            input_dim=vn['INPUT_DIM'],
            output_dim=vn['OUTPUT_DIM'],
            conv_hidden=vn['CONV_HIDDEN'],
            transformer_dim=vn['TRANSFORMER_DIM'],
            num_layers=vn['NUM_LAYERS'],
            activation=vn['ACTIVATION'],
            norm=vn['NORM'],
            dropout=vn['DROPOUT'],
            dim_feedforward=vn.get('DIM_FEEDFORWARD'),
            learnable_pe=vn.get('LEARNABLE_PE', True),
        )
    elif bb == 'bilstm':
        visual_net = ConvLSTM_VisualBiLSTM(
            input_dim=vn['INPUT_DIM'],
            output_dim=vn['OUTPUT_DIM'],
            conv_hidden=vn['CONV_HIDDEN'],
            lstm_hidden=vn['LSTM_HIDDEN'],
            num_layers=vn['NUM_LAYERS'],
            activation=vn['ACTIVATION'],
            norm=vn['NORM'],
            dropout=vn['DROPOUT'],
            pool_type=vn.get('POOL_TYPE', 'mean'),
        )
    else:
        raise ValueError(f"不支持的 VISUAL_NET.BACKBONE: {bb}，请使用 'transformer' 或 'bilstm'")

    evaluator = Evaluator(
        feature_dim=model_config['EVALUATOR']['INPUT_FEATURE_DIM'],
        predict_type=model_config['EVALUATOR']['PREDICT_TYPE'],
    )

    if len(args.gpu.split(',')) > 1:
        visual_net = nn.DataParallel(visual_net)
        evaluator = nn.DataParallel(evaluator)

    visual_net = visual_net.to(args.device)
    evaluator = evaluator.to(args.device)

    if model_config['WEIGHTS']['TYPE'].lower() == 'last':
        assert ckpt_path is not None, \
            "'ckpt_path' 必须提供给 'get_models' 函数 "
        weights_path = find_last_ckpts(path=ckpt_path,
                                       key=model_type,
                                       date=model_config['WEIGHTS']['DATE'])

    elif model_config['WEIGHTS']['TYPE'].lower() == 'absolute_path':
        assert model_config['WEIGHTS']['CUSTOM_ABSOLUTE_PATH'] is not None, \
            "'CUSTOM_ABSOLUTE_PATH' 权重文件的绝对路径不能在配置文件中为空"
        assert os.path.isabs(model_config['WEIGHTS']['CUSTOM_ABSOLUTE_PATH']), \
            "给定的 'CUSTOM_ABSOLUTE_PATH' 不是绝对路径"

        weights_path = str(model_config['WEIGHTS']['CUSTOM_ABSOLUTE_PATH'])

    elif model_config['WEIGHTS']['TYPE'].lower() != 'new':
        assert model_config['WEIGHTS']['NAME'] is not None, \
            "'NAME'（权重文件名）必须在配置文件中的 'WEIGHTS' 提供"
        weights_path = os.path.join(model_config['WEIGHTS']['PATH'], model_config['WEIGHTS']['NAME'])
    else:
        weights_path = None

    if weights_path is not None:
        model_config['WEIGHTS']['INCLUDED'] = [x.lower() for x in model_config['WEIGHTS']['INCLUDED']]
        checkpoint = torch.load(weights_path)

        if 'visual_net' in model_config['WEIGHTS']['INCLUDED']:
            print("正在加载视觉网络的权重： {}".format(weights_path))
            visual_net.load_state_dict(checkpoint['visual_net'])

        if 'evaluator' in model_config['WEIGHTS']['INCLUDED']:
            print("正在加载评估器的权重： {}".format(weights_path))
            evaluator.load_state_dict(checkpoint['evaluator'])

    return visual_net, evaluator


def save_model_weights(visual_net, evaluator, epoch, model_type, ckpt_path, best_metric=None, value=None):
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    checkpoint = {
        'epoch': epoch,
        'visual_net': visual_net.state_dict(),
        'evaluator': evaluator.state_dict(),
    }
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_filename = os.path.join(ckpt_path, f"{model_type}_{current_time}_best_{best_metric}={value}_epoch{epoch}.pth")

    torch.save(checkpoint, checkpoint_filename)
    print(f"模型检查点已保存: {checkpoint_filename}")


# 根据整个数据集计算每个类别交叉熵损失的权重
def get_crossentropy_weights_whole_data(data_config, evaluator_config):
    # 获取数据集的根目录路径
    root_dir = data_config['{}_ROOT_DIR'.format(data_config['MODE']).upper()]

    if evaluator_config['PREDICT_TYPE'] == 'pd-score':
        gt_path = os.path.join(root_dir, 'pd_score_gt.npy')
        gt = np.load(gt_path)

        # 初始化权重数组
        weights = np.zeros(evaluator_config['N_CLASSES'])
        # 获取每个标签的出现次数
        labels, counts = np.unique(gt, return_counts=True)
        for i in range(len(labels)):
            weights[labels[i]] = 1. / counts[i]

    elif evaluator_config['PREDICT_TYPE'] == 'pd-binary':
        gt_path = os.path.join(root_dir, 'pd_binary_gt.npy')
        gt = np.load(gt_path)

        # 初始化权重数组
        weights = np.zeros(evaluator_config['N_CLASSES'])
        # 获取每个标签的出现次数
        labels, counts = np.unique(gt, return_counts=True)
        for i in range(len(labels)):
            weights[labels[i]] = 1. / counts[i]
    
    else:
        raise AssertionError("未知的 'PREDICT_TYPE' 类型！", evaluator_config['PREDICT_TYPE'])

    return weights


# 根据给定的GT数据和评估器配置，计算交叉熵损失的权重。
# 这些权重可以用于平衡类别不均衡的问题。
def get_crossentropy_weights(gt, evaluator_config):
    if evaluator_config['PREDICT_TYPE'] == 'pd-score':

        weights = np.zeros(evaluator_config['N_CLASSES'])  # 初始化权重数组
        labels, counts = np.unique(gt, return_counts=True)  # 获取每个类别的样本数量
        for i in range(len(labels)):
            weights[int(labels[i])] = 1. / counts[i]  # 计算权重

    elif evaluator_config['PREDICT_TYPE'] == 'pd-binary':

        weights = np.zeros(evaluator_config['N_CLASSES'])  # 初始化权重数组
        labels, counts = np.unique(gt, return_counts=True)  # 获取每个类别的样本数量
        for i in range(len(labels)):
            weights[int(labels[i])] = 1. / counts[i]  # 计算权重
    
    else:
        raise AssertionError("未知的 'PREDICT_TYPE' 类型！", evaluator_config['PREDICT_TYPE'])

    return weights


# def get_criterion(criterion_config, args):
#     # 如果 USE_WEIGHTS 为 True，则使用配置中的权重来初始化 CrossEntropyLoss。
#     if criterion_config['USE_WEIGHTS']:
#
#         weights = torch.tensor(criterion_config['WEIGHTS']).type(torch.FloatTensor).to(args.device)
#         criterion = nn.CrossEntropyLoss(weight=weights)
#
#     else:
#         criterion = nn.CrossEntropyLoss()
#
#     return criterion

def get_criterion(criterion_config, args, samples_per_cls=None):
    if criterion_config.get('USE_CB_BCE_LOSS', False):
        return CB_Focal_BCEWithLogitsLoss(
            beta=criterion_config.get('CB_BETA', 0.999),
            samples_per_cls=samples_per_cls,
            gamma=criterion_config.get('CB_GAMMA', 2.0),
            device=args.device
        )
    elif criterion_config.get('USE_CB_LOSS', False):
        return ClassBalancedLoss(
            beta=criterion_config.get('CB_BETA', 0.999),
            samples_per_cls=samples_per_cls,
            num_classes=criterion_config['NUM_CLASSES'],
            loss_type=criterion_config.get('CB_LOSS_TYPE', 'cb_focal'),
            gamma=criterion_config.get('CB_GAMMA', 2.0),
            device=args.device
        )
    elif criterion_config.get('USE_WEIGHTS', False):
        weights = torch.tensor(criterion_config['WEIGHTS']).float().to(args.device)
        return nn.CrossEntropyLoss(weight=weights)
    else:
        return nn.CrossEntropyLoss()

# 根据配置获取优化器和学习率调度器。此函数可以选择使用标准的 Adam 优化器或 SAM (Sharpness-Aware Minimization) 优化器，并根据配置设置学习率调度器。
# def get_optimizer_scheduler(model_parameters, optimizer_config, scheduler_config):
#     # model_parameters：模型的参数
#     # optimizer_config：优化器的配置，包括学习率 (LR)、权重衰减 (WEIGHT_DECAY) 和是否使用 SAM (USE_SAM)。
#     # scheduler_config：调度器的配置，包括步长 (STEP_SIZE) 和学习率衰减因子 (GAMMA)。
#     if optimizer_config['USE_SAM']:
#         # 获取优化器和学习率调度器
#         # 如果 USE_SAM 为 False，则使用标准的 Adam 优化器，设置 lr（学习率）和 weight_decay（权重衰减）
#         base_optimizer = torch.optim.Adam
#         optimizer = SAM(model_parameters, base_optimizer, rho=2, adaptive=True, betas=(0.9, 0.999),
#                         lr=optimizer_config['LR'], weight_decay=optimizer_config['WEIGHT_DECAY'])
#
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer.base_optimizer,
#                                                     step_size=scheduler_config['STEP_SIZE'],
#                                                     gamma=scheduler_config['GAMMA'])
#     else:
#         optimizer = torch.optim.Adam(model_parameters, betas=(0.9, 0.999),
#                                      lr=optimizer_config['LR'],
#                                      weight_decay=optimizer_config['WEIGHT_DECAY'])
#
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
#                                                     step_size=scheduler_config['STEP_SIZE'],
#                                                     gamma=scheduler_config['GAMMA'])
#
#     return optimizer, scheduler
def get_optimizer_scheduler(model_parameters, optimizer_config, scheduler_config):
    """
    获取优化器与学习率调度器（支持 SAM / Lookahead / Adam / AdamW / SGD）

    参数说明：
    - model_parameters: 模型参数
    - optimizer_config: dict，包含：
        - TYPE: 优化器类型 ['Adam', 'AdamW', 'SGD']
        - USE_SAM: 是否启用 SAM
        - USE_LOOKAHEAD: 是否启用 Lookahead
        - LR: 学习率
        - WEIGHT_DECAY: 权重衰减
        - RHO: SAM的扰动半径
        - ADAPTIVE: SAM是否自适应
    - scheduler_config: dict，包含：
        - STEP_SIZE: 学习率步长
        - GAMMA: 学习率衰减系数
    """

    opt_type = optimizer_config.get('TYPE', 'Adam').lower()
    lr = optimizer_config['LR']
    wd = optimizer_config.get('WEIGHT_DECAY', 0)
    rho = optimizer_config.get('RHO', 0.05)
    adaptive = optimizer_config.get('ADAPTIVE', False)
    use_sam = optimizer_config.get('USE_SAM', False)
    use_lookahead = optimizer_config.get('USE_LOOKAHEAD', False)

    # === 选择基础优化器类 ===
    if opt_type == 'adam':
        base_opt_class = torch.optim.Adam
    elif opt_type == 'adamw':
        base_opt_class = torch.optim.AdamW
    elif opt_type == 'sgd':
        base_opt_class = torch.optim.SGD
    elif opt_type == 'radam':
        base_opt_class = optim_ext.RAdam  # 从 torch-optimizer 调用 RAdam
    elif opt_type == 'ranger':
        base_optimizer = optim_ext.Ranger(model_parameters,
                                          lr=lr,
                                          weight_decay=wd)
    else:
        raise ValueError(f"不支持的优化器类型: {opt_type}")

    # === 创建基础优化器 ===
    if opt_type == 'ranger':
        base_optimizer = optim_ext.Ranger(model_parameters, lr=lr, weight_decay=wd)
    elif opt_type == 'sgd':
        base_optimizer = base_opt_class(model_parameters, lr=lr, momentum=0.9, weight_decay=wd)
    else:
        base_optimizer = base_opt_class(model_parameters, lr=lr, betas=(0.9, 0.999), weight_decay=wd)

    # === 可选 Lookahead 包裹器 ===
    if use_lookahead:
        base_optimizer = Lookahead(base_optimizer)

    # === 是否启用 SAM ===
    if use_sam:
        optimizer = SAM(model_parameters,
                        base_optimizer=type(base_optimizer),
                        rho=rho,
                        adaptive=adaptive,
                        lr=lr,
                        weight_decay=wd,
                        betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer.base_optimizer,  # 注意：调度器作用在 base 上
                                                    step_size=scheduler_config['STEP_SIZE'],
                                                    gamma=scheduler_config['GAMMA'])
    else:
        optimizer = base_optimizer
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=scheduler_config['STEP_SIZE'],
                                                    gamma=scheduler_config['GAMMA'])

    return optimizer, scheduler








# 根据预测类型从数据中获取对应的真实标签 (ground truth) 数据
def get_gt(data, predict_type):
    """
    参数：
        data (dict): 包含不同真实标签数据的字典。
        predict_type (str): 预测类型，用于选择相应的真实标签数据。
    返回：
        gt: 根据预测类型选择的真实标签数据。
    """
    if predict_type == 'pd-score':
        gt = data['pd_score_gt']

    elif predict_type == 'pd-binary':
        gt = data['pd_binary_gt']
    
    else:
        # 如果预测类型不符合预期，抛出异常
        raise AssertionError("未知的 'PREDICT_TYPE' 类型！", predict_type)

    return gt


# 根据预测的概率分布（probs）计算最终的得分（score_pred） 并根据 evaluator_config 中的配置信息处理不同类型的预测
# 主要功能：将概率分布转换为实际的分数，用于评估模型的性能。
def compute_score(probs, evaluator_config, args):
    # probs: 模型输出的概率分布，可以是一个或多个子得分的概率分布。
    # evaluator_config: 配置字典，包含关于预测类型、类别数、分辨率等信息。
    # args: 参数对象，包含设备信息（如 device），用于将计算结果移动到合适的设备（如 GPU）
    score_pred = (probs.argmax(dim=-1)).to(float)

    return score_pred.to(args.device)


def compute_loss(criterion, logits, gt, args):
    gt = gt.float().to(args.device)  # BCE要求float
    if logits.dim() == 2 and logits.size(1) == 1:
        logits = logits.view(-1)  # [B, 1] → [B]
    return criterion(logits, gt)

# 用于生成标准格式的混淆矩阵。
# 这个函数的作用是接受真实标签和预测标签，计算并返回一个标准格式的混淆矩阵，
# 其中第一行包含真阳性和假阳性，第二行包含假阴性和真阴性。
def standard_confusion_matrix(gt, pred):
    """
    Make confusion matrix with format:
                  -----------
                  | TP | FP |
                  -----------
                  | FN | TN |
                  -----------
    Parameters
    ----------
    y_true : ndarray - 1D
    y_pred : ndarray - 1D

    Returns
    -------
    ndarray - 2D
    """
    [[tn, fp], [fn, tp]] = metrics.confusion_matrix(np.asarray(gt), np.asarray(pred))
    return np.array([[tp, fp], [fn, tn]])


# 计算分类模型的准确率和正确分类的样本数量。
def get_accuracy(gt, pred):
    [[tp, fp], [fn, tn]] = standard_confusion_matrix(gt, pred)
    accuracy = (tp + tn) / (tp + fp + fn + tn)  # 计算准确率：正确分类的样本数/所有样本的总数
    correct_number = tp + tn  # 计算正确分类的样本数量。
    return accuracy, correct_number


# 计算二分类模型的评估指标
def get_classification_scores(gt, pred):  # gt：真实值 pred：预测值
    [[tp, fp], [fn, tn]] = standard_confusion_matrix(gt, pred)  # 计算混淆矩阵
    # TPR(sensitivity), TNR(specificity)
    tpr = tp / (tp + fn)  # 计算真正率（True Positive Rate, TPR），也称为灵敏度（Sensitivity）
    tnr = tn / (tn + fp)  # 计算真负率（True Negative Rate, TNR），也称为特异度（Specificity）
    # Precision, Recall, F1-score
    precision = tp / (tp + fp)  # 计算精确率（Precision）
    recall = tp / (tp + fn)  # 计算召回率（Recall），即灵敏度（Sensitivity），与 TPR 相同
    f1_score = 2 * (precision * recall) / (precision + recall)  # 计算 F1 分数（F1-Score），它是精确率和召回率的调和平均值。
    return tpr, tnr, precision, recall, f1_score


# 计算回归模型预测结果的评估指标。
def get_regression_scores(gt, pred):  # gt：真实值 pred：预测值
    gt = np.array(gt).astype(float)
    pred = np.array(pred).astype(float)
    mae = metrics.mean_absolute_error(gt, pred)  # 平均绝对误差（Mean Absolute Error）
    mse = metrics.mean_squared_error(gt, pred)   # 均方误差（Mean Squared Error）
    rmse = np.sqrt(mse)  # or mse**(0.5)         # 均方根误差（Root Mean Squared Error）
    r2 = metrics.r2_score(gt, pred)              # 决定系数（Coefficient of Determination）
    return mae, mse, rmse, r2


# 记录并打印误分类的样本信息
def log_misclassified_samples(batch_size, pd_binary_pred, data, mode):
    """
    记录并打印误分类的样本信息。

    参数:
    - batch_size: 当前批次的样本数量
    - pd_binary_pred: 当前批次的预测结果
    - data: 当前批次的数据（包含ID和真实标签）

    返回:
    - misclassified_samples: 一个包含误分类样本信息的列表
    """
    misclassified_samples = []
    for i in range(batch_size):
        if pd_binary_pred[i] != data['pd_binary_gt'][i].item():  # 如果预测错误
            misclassified_samples.append({
                'ID': data['ID'][i].item(),
                '是否PD(真实)': data['pd_binary_gt'][i].item(),
                '是否PD(预测)': pd_binary_pred[i]
            })
    print("\n误分类样本:")
    for sample in misclassified_samples:  # 显示全部；显示前10个则使用misclassified_samples[:10]
        print(f"ID: {sample['ID']}, 是否PD(真实): {sample['是否PD(真实)']}, 是否PD(预测): {sample['是否PD(预测)']}")
    print(f"{mode.upper()} MODE: Finished processing")
    return misclassified_samples


# CBLoss 封装类（支持 CrossEntropy / Focal / CB-Focal 模式）
import torch.nn.functional as F
class ClassBalancedLoss(nn.Module):
    def __init__(self, beta, samples_per_cls, num_classes, loss_type='softmax', gamma=2.0, device='cpu'):
        """
        beta: Class-Balanced Loss 中的平衡因子，一般设置为 0.99 或 0.999
        samples_per_cls: 每个类别的样本数量（list 或 numpy）
        num_classes: 类别数
        loss_type: ['softmax', 'focal', 'cb_focal']
        gamma: focal loss 中的聚焦因子
        device: 'cuda' or 'cpu'
        """
        super(ClassBalancedLoss, self).__init__()
        self.beta = beta
        self.samples_per_cls = np.array(samples_per_cls)
        self.num_classes = num_classes
        self.loss_type = loss_type.lower()
        self.gamma = gamma
        self.device = device
        self.class_weights = self._compute_cb_weights()

    def _compute_cb_weights(self):
        # 这是原始 [CB-Loss 公式（Cui et al., 2019）]：
        # 样本数越少，effective_num 越小，权重越大
        # 正则化使权重总和 = 类别数
        effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
        weights = (1.0 - self.beta) / effective_num
        weights = weights / np.sum(weights) * self.num_classes  # normalize
        return torch.tensor(weights, dtype=torch.float32).to(self.device)

    def forward(self, logits, labels):
        # 获取每个样本的权重
        weights = self.class_weights[labels]

        if self.loss_type == 'softmax':
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)

        elif self.loss_type == 'focal':
            pred = F.softmax(logits, dim=1)
            ce_loss = F.cross_entropy(logits, labels, reduction='none')
            pt = pred[range(len(labels)), labels]
            focal_term = (1 - pt) ** self.gamma
            loss = (focal_term * ce_loss * weights).mean()

        elif self.loss_type == 'cb_focal':
            pred = F.softmax(logits, dim=1)
            ce_loss = F.cross_entropy(logits, labels, reduction='none')
            pt = pred[range(len(labels)), labels]
            focal_term = (1 - pt) ** self.gamma
            cb_weights = self.class_weights[labels]  # 每个样本的CB权重
            loss = (cb_weights * focal_term * ce_loss).mean()

        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

        return loss

# ----------------------------------------
# 二分类专用：BCE版 Class-Balanced Focal Loss
# ----------------------------------------
class CB_Focal_BCEWithLogitsLoss(nn.Module):
    def __init__(self, beta, samples_per_cls, gamma=2.0, device='cpu'):
        super(CB_Focal_BCEWithLogitsLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.device = device
        self.samples_per_cls = np.array(samples_per_cls)

        effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
        weights = (1.0 - self.beta) / effective_num
        weights = weights / np.sum(weights)
        self.class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

    def forward(self, logits, labels, pd_scores=None):
        # 强制 logits
        logits = torch.clamp(logits, min=-10, max=10)
        labels = labels.float()
        probs = torch.sigmoid(logits)
        # 防止 log(0)、除以0、(1-pt)^gamma = nan
        probs = torch.clamp(probs, min=1e-6, max=1.0 - 1e-6)

        # 根据标签选取正类或负类概率
        pt = torch.where(labels == 1, probs, 1 - probs)
        pt = torch.clamp(pt, min=1e-6, max=1.0 - 1e-6)  # 防止 pt 为 0

        # Focal 权重项
        focal_weight = (1 - pt) ** self.gamma
        # 类别平衡权重
        weights = torch.where(labels == 1, self.class_weights[1], self.class_weights[0])

        # # 加重轻症病人样本权重
        # if pd_scores is not None:
        #     mild_mask = (labels == 1) & (pd_scores <= 1.5)
        #     weights[mild_mask] *= 1.5

        # 使用 logits 计算 BCE
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        # 加权后返回平均损失
        loss = focal_weight * weights * bce_loss
        return loss.mean()

def get_samples_per_cls(gt, evaluator_config):
    samples_per_cls = [0] * evaluator_config['N_CLASSES']

    if evaluator_config['PREDICT_TYPE'] == 'pd-score':
        labels, counts = np.unique(gt, return_counts=True)
        for i in range(len(labels)):
            samples_per_cls[int(labels[i])] = counts[i]

    elif evaluator_config['PREDICT_TYPE'] == 'pd-binary':
        labels, counts = np.unique(gt, return_counts=True)
        for i in range(len(labels)):
            samples_per_cls[int(labels[i])] = counts[i]

    else:
        raise AssertionError("未知的 'PREDICT_TYPE' 类型！", evaluator_config['PREDICT_TYPE'])

    return samples_per_cls


EPOCH_TRACE_FIELDS = [
    "ts",
    "fold",
    "epoch",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "test_loss",
    "test_acc",
    "test_f1",
    "test_tpr",
    "test_tnr",
    "lr",
    "epoch_sec",
]


def write_run_manifest_json(path, meta):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def append_epoch_trace_row(csv_path, row, fieldnames=EPOCH_TRACE_FIELDS):
    file_exists = os.path.isfile(csv_path)
    out = {k: row.get(k, "") for k in fieldnames}
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            w.writeheader()
        w.writerow(out)


def write_run_session_summary_json(path, meta):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
