import os
import numpy as np
import torch
from torch.utils.data import Dataset

def pad_audio_feat(feat, target_len=300):
    """将特征帧数不足的补齐到固定长度"""
    C, T = feat.shape
    if T < target_len:
        pad_width = target_len - T
        pad = np.zeros((C, pad_width), dtype=feat.dtype)
        return np.concatenate((feat, pad), axis=1)
    else:
        return feat[:, :target_len]  # 若超过则裁剪


class PDDataset(Dataset):
    '''create a training, develop, or test dataset
       and load the participant features if it's called
    '''

    def __init__(self,
                 root_dir,
                 mode,
                 visual_with_face3d=True,
                 transform=None):
        super(PDDataset, self).__init__()
        
        self.mode = mode  # 模式，取值为 "train"、"validation" 或 "test"
        self.root_dir = root_dir  # 数据集的根目录
        self.visual_with_face3d = visual_with_face3d  # 是否使用眼动追踪数据
        self.transform = transform  # 用于数据转换的函数或方法

        # IDs、gender_gt 等参数是在 __init__ 方法中加载，而不通过参数传递，因为下面的这些参数需要yaml文件中的root_dir 和 mode 参数来确定数据集的路径
        self.IDs = np.load(os.path.join(self.root_dir, 'ID_gt.npy'))
        self.gender_gt = np.load(os.path.join(self.root_dir, 'gender_gt.npy'))
        self.pd_binary_gt = np.load(os.path.join(self.root_dir, 'pd_binary_gt.npy'))
        self.pd_score_gt = np.load(os.path.join(self.root_dir, 'pd_score_gt.npy'))

    def __len__(self):
        return len(self.IDs)

    def __iter__(self):
        return iter(self.IDs)

    # 从数据集中获取一个样本，并将其转换成一个字典形式的 session。
    def __getitem__(self, idx):
        '''
        Essentional function for creating dataset in PyTorch, which will automatically be
        called in Dataloader and load all the extracted features of the patient in the Batch
        based on the index of self.IDs
        Argument:
            idx: int, index of the patient ID in self.IDs
        Return:
            session: dict, contains all the extracted features and ground truth of a patient/session
        '''

        # 如果 idx 是一个张量，则将其转换为列表
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 使用 face3d 数据和不使用 face3d 数据
        if self.visual_with_face3d:
            fkps_path = os.path.join(self.root_dir, 'facial_keypoints')
            aus_path = os.path.join(self.root_dir, 'action_units')
            pos_path = os.path.join(self.root_dir, 'position_rotation')

            fkps_file = np.sort(os.listdir(fkps_path))[idx]
            aus_file = np.sort(os.listdir(aus_path))[idx]
            pos_file = np.sort(os.listdir(pos_path))[idx]

            fkps = np.load(os.path.join(fkps_path, fkps_file))  # (780, 68, 3)
            aus = np.load(os.path.join(aus_path, aus_file))  # (780, 17)
            # print("Shape of aus:", aus.shape)
            pos = np.load(os.path.join(pos_path, pos_file))  # (780, 2, 3)
            # 将 2D特征aus 扩展到帧级别
            aus_3d = np.expand_dims(aus, axis=2).repeat(3, axis=2)  # (780, 16, 3)

            # 拼接所有 3D 特征
            visual = np.concatenate((fkps, pos, aus_3d), axis=1)  # (780, 87, 3)
            # print("Shape of visual:", visual.shape)  # (300, 87, 3)


        else:
            # fkps_path = os.path.join(self.root_dir, 'facial_keypoints')
            aus_path = os.path.join(self.root_dir, 'action_units')
            pos_path = os.path.join(self.root_dir, 'position_rotation')
            # fkps_file = np.sort(os.listdir(fkps_path))[idx]
            aus_file = np.sort(os.listdir(aus_path))[idx]
            pos_file = np.sort(os.listdir(pos_path))[idx]

            aus = np.load(os.path.join(aus_path, aus_file))  # (150, 14)
            pos = np.load(os.path.join(pos_path, pos_file))  # (150, 2, 3)
            # 将 2D特征aus 扩展到帧级别
            aus_3d = np.expand_dims(aus, axis=2).repeat(3, axis=2)  # (150, 14, 3)

            # 拼接所有 3D 特征
            visual = np.concatenate((pos, aus_3d), axis=1)  # (150, 16, 3)
            # print('Size of visual', visual.shape)


        # 加载音频特征
        audio_path = os.path.join(self.root_dir, 'audio')

        # 二维
        mel_path = os.path.join(audio_path, 'mel_spectrogram')
        log_mel_path = os.path.join(audio_path, 'log_mel_spectrogram')
        delta_log_path = os.path.join(audio_path, 'delta_log_mel')
        delta2_log_path = os.path.join(audio_path, 'delta2_log_mel')

        mfcc_path = os.path.join(audio_path, 'mfcc')
        delta_path = os.path.join(audio_path, 'delta_mfcc')
        delta2_path = os.path.join(audio_path, 'delta2_mfcc')

        centroid_path = os.path.join(audio_path, 'spectral_centroid')
        bandwidth_path = os.path.join(audio_path, 'spectral_bandwidth')
        contrast_path = os.path.join(audio_path, 'spectral_contrast')
        rolloff_path = os.path.join(audio_path, 'spectral_rolloff')
        zero_crossing_rate_path = os.path.join(audio_path, 'zero_crossing_rate')


        # 标量/一维医学音频特征路径
        fo_path = os.path.join(audio_path, 'MDVP_Fo')
        fhi_path = os.path.join(audio_path, 'MDVP_Fhi')
        flo_path = os.path.join(audio_path, 'MDVP_Flo')
        jitter_path = os.path.join(audio_path, 'Jitter_percent')
        jitter_abs_path = os.path.join(audio_path, 'Jitter_abs')
        shimmer_path = os.path.join(audio_path, 'Shimmer')
        shimmer_db_path = os.path.join(audio_path, 'Shimmer_dB')
        hnr_path = os.path.join(audio_path, 'HNR')
        nhr_path = os.path.join(audio_path, 'NHR')
        # rpde_path = os.path.join(audio_path, 'RPDE')
        # dfa_path = os.path.join(audio_path, 'DFA')
        # d2_path = os.path.join(audio_path, 'D2')

        # 获取音频文件
        mel_file = np.sort(os.listdir(mel_path))[idx]
        log_mel_file = np.sort(os.listdir(log_mel_path))[idx]
        delta_log_file = np.sort(os.listdir(delta_log_path))[idx]
        delta2_log_file = np.sort(os.listdir(delta2_log_path))[idx]

        mfcc_file = np.sort(os.listdir(mfcc_path))[idx]
        delta_file = np.sort(os.listdir(delta_path))[idx]
        delta2_file = np.sort(os.listdir(delta2_path))[idx]

        centroid_file = np.sort(os.listdir(centroid_path))[idx]
        bandwidth_file = np.sort(os.listdir(bandwidth_path))[idx]
        contrast_file = np.sort(os.listdir(contrast_path))[idx]
        rolloff_file = np.sort(os.listdir(rolloff_path))[idx]
        zero_crossing_rate_file = np.sort(os.listdir(zero_crossing_rate_path))[idx]

        fo_file = np.sort(os.listdir(fo_path))[idx]
        fhi_file = np.sort(os.listdir(fhi_path))[idx]
        flo_file = np.sort(os.listdir(flo_path))[idx]
        jitter_file = np.sort(os.listdir(jitter_path))[idx]
        jitter_abs_file = np.sort(os.listdir(jitter_abs_path))[idx]
        shimmer_file = np.sort(os.listdir(shimmer_path))[idx]
        shimmer_db_file = np.sort(os.listdir(shimmer_db_path))[idx]
        hnr_file = np.sort(os.listdir(hnr_path))[idx]
        nhr_file = np.sort(os.listdir(nhr_path))[idx]
        # rpde_file = np.sort(os.listdir(rpde_path))[idx]
        # dfa_file = np.sort(os.listdir(dfa_path))[idx]
        # d2_file = np.sort(os.listdir(d2_path))[idx]

        # 加载音频特征(二维)
        # Spectral Bandwidth、Spectral Centroid、Spectral、Rolloff、Zero Crossing Rate

        mel_spectrogram = np.load(os.path.join(mel_path, mel_file))
        # print('Size of mel_spectrogram', mel_spectrogram.shape)  # (128, 300)
        log_mel_spectrogram = np.load(os.path.join(log_mel_path, log_mel_file))
        # print('Size of log_mel_spectrogram', log_mel_spectrogram.shape)  # (128, 300)
        delta_log_mel = np.load(os.path.join(delta_log_path, delta_log_file))
        # print('Size of delta_log_mel', delta_log_mel.shape)  # (128, 300)
        delta2_log_mel = np.load(os.path.join(delta2_log_path, delta2_log_file))
        # print('Size of delta2_log_mel', delta2_log_mel.shape)  # (128, 300)

        mfcc = np.load(os.path.join(mfcc_path, mfcc_file))  # (20, 300)
        # print('Size of mfcc', mfcc.shape)  # (20, 300)
        delta = np.load(os.path.join(delta_path, delta_file))  # (20, 300)
        # print('Size of delta', delta.shape)  # (20, 300)
        delta2 = np.load(os.path.join(delta2_path, delta2_file))  # (20, 300)
        # print('Size of delta2', delta2.shape)  # (20, 300)
        bandwidth = np.load(os.path.join(bandwidth_path, bandwidth_file))  # (1, 300)
        # print('Size of bandwidth', bandwidth.shape)  # (1, 300)
        centroid = np.load(os.path.join(centroid_path, centroid_file))  # (1, 300)
        # print('Size of centroid', centroid.shape)  # (1, 300)
        contrast = np.load(os.path.join(contrast_path, contrast_file))  #  (7, 300)
        # print('Size of contrast', contrast.shape)  #  (7, 300)
        rolloff = np.load(os.path.join(rolloff_path, rolloff_file))  # (1, 300)
        # print('Size of rolloff', rolloff.shape)  # (1, 300)
        zero_crossing_rate = np.load(os.path.join(zero_crossing_rate_path, zero_crossing_rate_file))  # (1, 300)
        # print('Size of zero_crossing_rate', zero_crossing_rate.shape)  # (1, 300)


        # 加载音频特征(标量)
        fo = np.load(os.path.join(fo_path, fo_file)).item() if np.load(
            os.path.join(fo_path, fo_file)).ndim == 0 else np.load(os.path.join(fo_path, fo_file))
        fhi = np.load(os.path.join(fhi_path, fhi_file)).item()
        flo = np.load(os.path.join(flo_path, flo_file)).item()
        jitter = np.load(os.path.join(jitter_path, jitter_file)).item()
        jitter_abs = np.load(os.path.join(jitter_abs_path, jitter_abs_file)).item()
        shimmer = np.load(os.path.join(shimmer_path, shimmer_file)).item()
        shimmer_db = np.load(os.path.join(shimmer_db_path, shimmer_db_file)).item()
        hnr = np.load(os.path.join(hnr_path, hnr_file)).item()
        nhr = np.load(os.path.join(nhr_path, nhr_file)).item()
        # rpde = np.load(os.path.join(rpde_path, rpde_file)).item()
        # dfa = np.load(os.path.join(dfa_path, dfa_file)).item()
        # d2 = np.load(os.path.join(d2_path, d2_file)).item()

        # 构建标量特征矩阵：每个特征扩展成 (1, num_frames)
        num_frames = 300
        audio_scalar_stack = np.vstack([
            np.full((1, num_frames), fo),
            # np.full((1, num_frames), fhi),
            # np.full((1, num_frames), flo),
            np.full((1, num_frames), jitter),
            # np.full((1, num_frames), jitter_abs),
            np.full((1, num_frames), shimmer),
            # np.full((1, num_frames), shimmer_db),
            np.full((1, num_frames), hnr),
            # np.full((1, num_frames), nhr)
            # np.full((1, num_frames), rpde),
            # np.full((1, num_frames), dfa),
            # np.full((1, num_frames), d2)
        ])  # shape: (12, num_frames)

        # 拼接音频特征
        # audio = np.concatenate((delta_log_mel, delta2_log_mel, log_mel_spectrogram, mfcc, delta, delta2, centroid, rolloff, zero_crossing_rate), axis=0)  # (447, 300)
        # audio = np.concatenate((delta_log_mel, delta2_log_mel, log_mel_spectrogram, centroid, zero_crossing_rate), axis=0)  # (386, 300)
        audio = np.concatenate((mfcc, delta, delta2),axis=0)  # (39, 900)
        # print('Size of audio',audio.shape)  # (39, 900)

        # summary
        # 生成包含不同键的字典
        session = {'ID': self.IDs[idx],
                    'gender_gt': self.gender_gt[idx],
                    'pd_binary_gt': self.pd_binary_gt[idx],
                    'pd_score_gt': self.pd_score_gt[idx],
                    'visual': visual,
                    'audio': audio,
                   }

        if self.transform:
            session = self.transform(session)

        return session

# 将 session 中的 NumPy 数组和整数类型转换为 PyTorch 的张量（Tensor） 以便后续操作能够在 GPU 上进行高效计算。
class ToTensor(object):
    # 核心方法，将 session 中的数据转换为 PyTorch 张量
    def __call__(self, session):
        converted_session = {'ID': session['ID'],
                            'gender_gt': torch.tensor(session['gender_gt']).type(torch.FloatTensor),
                            'pd_binary_gt': torch.tensor(session['pd_binary_gt']).type(torch.FloatTensor),
                            'pd_score_gt': torch.tensor(session['pd_score_gt']).type(torch.FloatTensor),
                            'visual': torch.from_numpy(session['visual']).type(torch.FloatTensor),
                            'audio': torch.from_numpy(session['audio']).type(torch.FloatTensor),}
        
        return converted_session


# 通过加载和处理数据集来演示如何使用 DataLoader 和 WeightedRandomSampler 来平衡数据集中的类分布
if __name__ == '__main__':
    from torch.utils.data import WeightedRandomSampler, DataLoader
    from torchvision import transforms

    root_dir = ('E:/Graduation matters/数据集/预处理-10+2不去噪/test')

    # test 3: try to load the dataset with DataLoader
    # 创建并转换数据集
    transformed_dataset = PDDataset(root_dir, 'train',
                                            transform=transforms.Compose([ToTensor()]))

    # show pd binary distribution
    pd_binary_gt = transformed_dataset.pd_binary_gt
    print('target train 0/1: {}/{}'.format(len(np.where(pd_binary_gt == 0)[0]), len(np.where(pd_binary_gt == 1)[0])))

    # show pd score distribution
    # 显示 PD 二元分类和分数分布
    pd_score_gt = transformed_dataset.pd_score_gt
    class_sample_ID, class_sample_count = np.unique(pd_score_gt, return_counts=True)
    print('class_sample_ID   : {}'.format(class_sample_ID))
    print('class_sample_count: {}'.format(class_sample_count))
    print('='*90)

    # 计算样本权重
    weight = 1. / class_sample_count
    samples_weight = np.zeros(pd_score_gt.shape)
    for i, sample_id in enumerate(class_sample_ID):
        indices = np.where(pd_score_gt == sample_id)[0]
        value = weight[i]
        samples_weight[indices] = value

    # samples_weight = weight[pd_binary_gt]
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    # DataLoader 加载数据集，并应用上面创建的 sampler 来平衡采样
    # create dataloader
    dataloader = DataLoader(transformed_dataset,
                            batch_size=20,
                            num_workers=1,
                            sampler=sampler)


    # iterate through batches
    # 迭代批次并打印信息
    total_count = np.zeros(class_sample_ID.shape)
    for i_batch, sample_batched in enumerate(dataloader):
        print('Batch number: ', i_batch, ', audio: ', sample_batched['audio'].size())
        num_count = []
        for id in class_sample_ID:
            num_count.append(len(np.where(sample_batched['pd_score_gt'].numpy() == id)[0]))
        print('loaded data PD Score Classes     : {}'.format(class_sample_ID))
        print('loaded data PD Score Distribution: {}'.format(num_count))
        # print('Participant IDs: {}'.format(sample_batched['ID']))
        print('='*90)
        total_count += num_count
    print('Total chosen classes: {}'.format(class_sample_ID))
    print('Amount of each class: {}'.format(total_count))

   