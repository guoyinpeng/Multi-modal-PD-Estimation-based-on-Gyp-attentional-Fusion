"""
多模态特征预处理（滑窗落盘 + GT 标签）

与 `models/AV-attention/utils.py` 中五折交叉验证的约定：
- `get_dataloaders` 使用 `StratifiedGroupKFold(..., groups=ids)`，`ids` 来自每个样本的受试者 ID；
- `_extract_ids_and_labels` 使用 `int(sample['ID'])`，且 `PDDataset` 用 `np.sort(os.listdir(子目录))[idx]`
  对齐各模态文件，因此：
  1) 所有窗口文件名必须带**固定宽度**的受试者编号，使字典序 = 数值序（如 000042-03_*.npy）；
  2) `ID_gt.npy` 等与 `facial_keypoints` 下 `np.sort` 顺序一致，且每条为**同一受试者**的整数 ID；
  3) 音频特征时间轴与视觉帧数不同，不能再用视觉帧下标直接切 `feature[:, start:end]`，需按时间对齐并插值到窗口长度。

占位：`audio/` 下医学标量（MDVP_* 等）若暂未从 Praat 提取，保存为 0.0，保证与 `dataset.py` 路径一致。
"""

from __future__ import annotations

import argparse
import os
import struct
from typing import Any, Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd

AUDIO_N_FFT = 2048
AUDIO_HOP = 533
N_MFCC = 13
N_MELS = 128

def create_folders(root_dir: str) -> None:
    """与 `dataset.py` 中 `PDDataset` 加载路径一致（含 log_mel、delta_log、标量医学特征）。"""
    subs = [
        "facial_keypoints",
        "gaze_vectors",
        "action_units",
        "position_rotation",
        "hog_features",
    ]
    audio_subs = [
        "mel_spectrogram",
        "log_mel_spectrogram",
        "delta_log_mel",
        "delta2_log_mel",
        "mfcc",
        "delta_mfcc",
        "delta2_mfcc",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_contrast",
        "spectral_rolloff",
        "zero_crossing_rate",
        "MDVP_Fo",
        "MDVP_Fhi",
        "MDVP_Flo",
        "Jitter_percent",
        "Jitter_abs",
        "Shimmer",
        "Shimmer_dB",
        "HNR",
        "NHR",
    ]
    os.makedirs(root_dir, exist_ok=True)
    for j in subs:
        os.makedirs(os.path.join(root_dir, j), exist_ok=True)
    for m in audio_subs:
        os.makedirs(os.path.join(root_dir, "audio", m), exist_ok=True)


def min_max_scaler(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    mn, mx = float(np.nanmin(data)), float(np.nanmax(data))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < 1e-12:
        return np.zeros_like(data, dtype=np.float32)
    return ((data - mn) / (mx - mn)).astype(np.float32)


def pre_check(data_df: pd.DataFrame) -> pd.DataFrame:
    data_df = data_df.apply(pd.to_numeric, errors="coerce")
    rest = data_df.iloc[:, 4:].to_numpy()
    valid = rest[~np.isnan(rest)]
    fill = 0.0 if valid.size == 0 else float(np.nanmin(valid))
    return data_df.where(~np.isnan(data_df), fill)


def load_gaze(gaze_path: str) -> np.ndarray:
    gaze_df = pre_check(pd.read_csv(gaze_path, low_memory=False))
    gaze_coor = gaze_df.iloc[:, 4:].to_numpy().reshape(len(gaze_df), 4, 3)
    return gaze_coor.astype(np.float32)


def load_keypoints(keypoints_path: str) -> np.ndarray:
    fkps_df = pre_check(pd.read_csv(keypoints_path, low_memory=False))
    x_coor = min_max_scaler(fkps_df[fkps_df.columns[4:72]].to_numpy())
    y_coor = min_max_scaler(fkps_df[fkps_df.columns[72:140]].to_numpy())
    z_coor = min_max_scaler(fkps_df[fkps_df.columns[140:208]].to_numpy())
    fkps_coor = np.stack([x_coor, y_coor, z_coor], axis=-1)
    return fkps_coor.astype(np.float32)


def load_AUs(AUs_path: str) -> np.ndarray:
    AUs_df = pre_check(pd.read_csv(AUs_path, low_memory=False))
    n = AUs_df.shape[1] - 4
    raw = AUs_df.iloc[:, 4 : 4 + min(17, n)].to_numpy()
    scaled = min_max_scaler(raw)
    if scaled.shape[1] < 17:
        pad = np.zeros((scaled.shape[0], 17 - scaled.shape[1]), dtype=np.float32)
        scaled = np.hstack([scaled.astype(np.float32), pad])
    return scaled.astype(np.float32)


def load_pose(pose_path: str) -> np.ndarray:
    pose_df = pre_check(pd.read_csv(pose_path, low_memory=False))
    pose_coor = pose_df.iloc[:, 4:].to_numpy()
    T, C = pose_coor.shape
    pose_features = np.zeros((T, C), dtype=np.float32)
    pose_features[:, :3] = min_max_scaler(pose_coor[:, :3])
    pose_features[:, 3:] = pose_coor[:, 3:].astype(np.float32)
    return pose_features.reshape(T, 2, 3)


def read_hog(filename: str, batch_size: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    all_feature_vectors: List[np.ndarray] = []
    with open(filename, "rb") as f:
        num_cols, = struct.unpack("i", f.read(4))
        num_rows, = struct.unpack("i", f.read(4))
        num_channels, = struct.unpack("i", f.read(4))
        num_features = 1 + num_rows * num_cols * num_channels
        feature_vector = struct.unpack("{}f".format(num_features), f.read(num_features * 4))
        feature_vector = np.array(feature_vector).reshape((1, num_features))
        all_feature_vectors.append(feature_vector)
        num_floats_per_feature_vector = 4 + num_rows * num_cols * num_channels
        num_floats_to_read = num_floats_per_feature_vector * batch_size
        num_bytes_to_read = num_floats_to_read * 4
        while True:
            blob = f.read(num_bytes_to_read)
            num_bytes_read = len(blob)
            if num_bytes_read == 0:
                break
            assert num_bytes_read % 4 == 0
            num_floats_read = num_bytes_read // 4
            assert num_floats_read % num_floats_per_feature_vector == 0
            num_feature_vectors_read = num_floats_read // num_floats_per_feature_vector
            feature_vectors = struct.unpack("{}f".format(num_floats_read), blob)
            feature_vectors = np.array(feature_vectors).reshape(
                (num_feature_vectors_read, num_floats_per_feature_vector)
            )
            feature_vectors = feature_vectors[:, 3:]
            all_feature_vectors.append(feature_vectors)
            if num_bytes_read < num_bytes_to_read:
                break
    all_feature_vectors = np.concatenate(all_feature_vectors, axis=0)
    is_valid = all_feature_vectors[:, 0]
    feature_vectors = all_feature_vectors[:, 1:]
    return is_valid, feature_vectors


def load_hog(hog_path: str) -> np.ndarray:
    _, hog_features = read_hog(hog_path)
    return hog_features.astype(np.float32)


def interp_time_axis(x: np.ndarray, n_out: int) -> np.ndarray:
    """将 (C, T_in) 插值为 (C, n_out)，与视觉窗口帧数一致。"""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"expected 2D audio feature, got {x.shape}")
    if x.shape[1] == n_out:
        return x.astype(np.float32)
    t_in = x.shape[1]
    if t_in == 0:
        return np.zeros((x.shape[0], n_out), dtype=np.float32)
    xp = np.linspace(0.0, 1.0, t_in)
    xnew = np.linspace(0.0, 1.0, n_out)
    y = np.zeros((x.shape[0], n_out), dtype=np.float32)
    for i in range(x.shape[0]):
        y[i] = np.interp(xnew, xp, x[i].astype(np.float64)).astype(np.float32)
    return y


def normalize_max_abs(x: np.ndarray) -> np.ndarray:
    m = np.max(np.abs(x))
    if m < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x / m).astype(np.float32)


def extract_audio_features_segment(
    y: np.ndarray,
    sr: int,
    n_out: int,
) -> Dict[str, np.ndarray]:
    """
    对一段波形提取与 `dataset.py` 子目录名一致的各特征，并插值到 n_out 列（与视觉窗口长度一致）。
    """
    y = np.asarray(y, dtype=np.float32)
    if y.size < AUDIO_N_FFT:
        y = np.pad(y, (0, AUDIO_N_FFT - len(y)), mode="constant")

    mel_power = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=AUDIO_N_FFT,
        hop_length=AUDIO_HOP,
        n_mels=N_MELS,
    )
    log_mel = librosa.power_to_db(mel_power)
    delta_log = librosa.feature.delta(log_mel)
    delta2_log = librosa.feature.delta(log_mel, order=2)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=AUDIO_N_FFT,
        hop_length=AUDIO_HOP,
        n_mels=N_MELS,
    )
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=AUDIO_N_FFT, hop_length=AUDIO_HOP
    )
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=AUDIO_N_FFT, hop_length=AUDIO_HOP
    )
    spectral_contrast = librosa.feature.spectral_contrast(S=mel_power, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=AUDIO_N_FFT, hop_length=AUDIO_HOP
    )
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y, hop_length=AUDIO_HOP)

    raw: Dict[str, np.ndarray] = {
        "mel_spectrogram": mel_power,
        "log_mel_spectrogram": log_mel,
        "delta_log_mel": delta_log,
        "delta2_log_mel": delta2_log,
        "mfcc": mfcc,
        "delta_mfcc": delta_mfcc,
        "delta2_mfcc": delta2_mfcc,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_contrast": spectral_contrast,
        "spectral_rolloff": spectral_rolloff,
        "zero_crossing_rate": zero_crossing_rate,
    }

    out: Dict[str, np.ndarray] = {}
    for k, v in raw.items():
        out[k] = interp_time_axis(v.astype(np.float32), n_out)
        out[k] = normalize_max_abs(out[k])
    return out


def sliding_window_ranges(T: int, frame_size: int, hop_size: int) -> List[Tuple[int, int]]:
    """与「仅完整窗口」一致：最后一窗不足则丢弃。"""
    out: List[Tuple[int, int]] = []
    start = 0
    while start + frame_size <= T:
        out.append((start, start + frame_size))
        start += hop_size
    return out


def save_scalar_placeholder(root: str, rel_audio_sub: str, fname: str, value: float = 0.0) -> None:
    path = os.path.join(root, "audio", rel_audio_sub, fname)
    np.save(path, np.float32(value))


def sliding_window(
    fkps_features: np.ndarray,
    gaze_features: np.ndarray,
    AUs_features: np.ndarray,
    pose_features: np.ndarray,
    hog_features: np.ndarray,
    y_audio: np.ndarray,
    audio_sr: int,
    visual_sr: int,
    window_size: float,
    overlap_size: float,
    output_root: str,
    file_tag: str,
) -> int:
    """
    file_tag: 固定宽度受试者编号字符串，如 '000042'，窗口文件名为 `{file_tag}-{wi:02}_*.npy`。
    """
    T = fkps_features.shape[0]
    frame_size = int(round(window_size * visual_sr))
    hop_size = int(round((window_size - overlap_size) * visual_sr))
    if T < frame_size or hop_size <= 0:
        return 0

    segs = sliding_window_ranges(T, frame_size, hop_size)
    for wi, (start, end) in enumerate(segs):
        n_vis = end - start
        prefix = f"{file_tag}-{wi:02d}"

        np.save(
            os.path.join(output_root, "facial_keypoints", f"{prefix}_kps.npy"),
            fkps_features[start:end],
        )
        np.save(
            os.path.join(output_root, "gaze_vectors", f"{prefix}_gaze.npy"),
            gaze_features[start:end],
        )
        np.save(
            os.path.join(output_root, "action_units", f"{prefix}_AUs.npy"),
            AUs_features[start:end],
        )
        np.save(
            os.path.join(output_root, "position_rotation", f"{prefix}_pose.npy"),
            pose_features[start:end],
        )
        np.save(
            os.path.join(output_root, "hog_features", f"{prefix}_hog.npy"),
            hog_features[start:end],
        )

        t0 = start / float(visual_sr)
        t1 = end / float(visual_sr)
        i0 = max(0, int(np.floor(t0 * audio_sr)))
        i1 = min(len(y_audio), int(np.ceil(t1 * audio_sr)))
        seg = y_audio[i0:i1].astype(np.float32)
        if seg.size == 0:
            seg = np.zeros(AUDIO_N_FFT, dtype=np.float32)

        audio_feats = extract_audio_features_segment(seg, int(audio_sr), n_vis)
        for name, arr in audio_feats.items():
            np.save(
                os.path.join(output_root, "audio", name, f"{prefix}_{name}.npy"),
                arr,
            )

        # 与 dataset 中 MDVP_* 等标量路径一致：占位标量（单值 .npy）
        scalar_keys = [
            "MDVP_Fo",
            "MDVP_Fhi",
            "MDVP_Flo",
            "Jitter_percent",
            "Jitter_abs",
            "Shimmer",
            "Shimmer_dB",
            "HNR",
            "NHR",
        ]
        for sk in scalar_keys:
            save_scalar_placeholder(output_root, sk, f"{prefix}_{sk}.npy", 0.0)

    return len(segs)


def main() -> None:
    parser = argparse.ArgumentParser(description="PD-HAUST 滑窗预处理（五折 CV 对齐）")
    parser.add_argument("--data_root", type=str, required=True, help="原始数据根目录（每受试者子文件夹）")
    parser.add_argument("--output_root", type=str, required=True, help="输出 train 根目录（含各模态子文件夹）")
    parser.add_argument("--csv", type=str, required=True, help="含 Participant_ID, PD_Binary, PD_Score, Gender 等列")
    parser.add_argument("--window", type=float, default=10.0)
    parser.add_argument("--overlap", type=float, default=2.0)
    parser.add_argument("--visual_sr", type=int, default=30)
    args = parser.parse_args()

    create_folders(args.output_root)

    gt_df = pd.read_csv(args.csv)
    if "Participant_ID" not in gt_df.columns:
        raise ValueError("CSV 必须包含 Participant_ID 列")

    # 按受试者 ID 数值排序处理，且文件名使用定宽编号，保证与 np.sort(os.listdir) 一致
    gt_df = gt_df.copy()
    gt_df["_pid_int"] = gt_df["Participant_ID"].apply(lambda x: int(str(x).strip()))
    gt_df = gt_df.sort_values("_pid_int").reset_index(drop=True)
    dup = gt_df["_pid_int"].duplicated()
    if dup.any():
        raise ValueError(f"CSV 中存在重复 Participant_ID: {gt_df.loc[dup, '_pid_int'].tolist()}")

    GT: Dict[str, List[Any]] = {
        "ID_gt": [],
        "gender_gt": [],
        "pd_binary_gt": [],
        "pd_score_gt": [],
        "pd_subscores_gt": [],
    }

    for i in range(len(gt_df)):
        patient_id_int = int(gt_df["_pid_int"].iloc[i])
        file_tag = f"{patient_id_int:06d}"
        pid = str(gt_df["Participant_ID"].iloc[i]).strip()
        pd_binary_gt = gt_df["PD_Binary"].iloc[i]
        pd_score_gt = gt_df["PD_Score"].iloc[i]
        gender_gt = gt_df["Gender"].iloc[i]
        row_no_pid = gt_df.drop(columns=["_pid_int"]).iloc[i]
        pd_subscores_gt = row_no_pid.iloc[4:].to_numpy().tolist()

        print(f"Processing Participant {pid} ({file_tag}) ...")

        base = os.path.join(args.data_root, pid)
        keypoints_path = os.path.join(base, f"{pid}_CLNF_3D.txt")
        gaze_path = os.path.join(base, f"{pid}_CLNF_gaze.txt")
        AUs_path = os.path.join(base, f"{pid}_CLNF_AUs.txt")
        pose_path = os.path.join(base, f"{pid}_CLNF_pose.txt")
        hog_path = os.path.join(base, f"{pid}_CLNF_hog.bin")
        audio_path = os.path.join(base, f"{pid}_AUDIO.wav")

        fkps_features = load_keypoints(keypoints_path)
        gaze_features = load_gaze(gaze_path)
        AUs_features = load_AUs(AUs_path)
        pose_features = load_pose(pose_path)
        hog_features = load_hog(hog_path)

        y_audio, audio_sr = librosa.load(audio_path, sr=None, mono=True)

        T = fkps_features.shape[0]
        if (
            gaze_features.shape[0] != T
            or AUs_features.shape[0] != T
            or pose_features.shape[0] != T
            or hog_features.shape[0] != T
        ):
            raise ValueError(
                f"受试者 {pid} 各模态帧数不一致: fkps {T}, gaze {gaze_features.shape[0]}, "
                f"aus {AUs_features.shape[0]}, pose {pose_features.shape[0]}, hog {hog_features.shape[0]}"
            )

        num_frame = sliding_window(
            fkps_features,
            gaze_features,
            AUs_features,
            pose_features,
            hog_features,
            y_audio,
            int(audio_sr),
            args.visual_sr,
            args.window,
            args.overlap,
            args.output_root,
            file_tag,
        )

        for _ in range(num_frame):
            GT["ID_gt"].append(patient_id_int)
            GT["gender_gt"].append(gender_gt)
            GT["pd_binary_gt"].append(pd_binary_gt)
            GT["pd_score_gt"].append(pd_score_gt)
            GT["pd_subscores_gt"].append(pd_subscores_gt)

    for k, v in GT.items():
        if k == "pd_subscores_gt":
            np.save(os.path.join(args.output_root, f"{k}.npy"), np.array(v, dtype=object))
        else:
            np.save(os.path.join(args.output_root, f"{k}.npy"), np.array(v))

    print("All done. 请确认 facial_keypoints 下 np.sort 顺序与 GT 行顺序一致（已定宽文件名 + 按 ID 排序处理）。")


if __name__ == "__main__":
    main()
