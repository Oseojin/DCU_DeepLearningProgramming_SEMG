"""윈도우를 만들기 전에 CSV(시행) 단위로 먼저 분할한다. 시행 정체성을 끝까지 보존한다."""

import hashlib
from pathlib import Path

import numpy as np
import pywt
import torch
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import TensorDataset

SUBJECTS = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data"
FS = 1000
WINDOW = 300
EVAL_HOP = 150          # 논문과 동일한 50% 중첩. 평가는 항상 이 값을 쓴다.
SCALES = list(range(1, 33))
WAVELET = "morl"

PREPROCESSING = {
    "fs": FS, "window": WINDOW, "eval_hop": EVAL_HOP, "scales": SCALES,
    "wavelet": WAVELET, "normalization": "per_window_joint_channel_minmax",
    "third_channel": "mean_of_two_absolute_cwt_maps", "input_shape": [3, 32, WINDOW],
}

_CACHE = {}


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_manifest(data_dir, split_seed=42, folds=5):
    root = Path(data_dir).resolve()
    records = []
    for path in sorted(root.rglob("*.csv")):
        if path.parent.name not in SUBJECTS:
            raise ValueError(f"CSV parent directory must be A-E: {path}")
        records.append({
            "path": path.relative_to(root).as_posix(),
            "label": SUBJECTS[path.parent.name], "sha256": file_digest(path),
        })
    if {r["label"] for r in records} != set(SUBJECTS.values()):
        raise ValueError(f"All five subjects A-E are required under {root}")
    development, test = train_test_split(
        records, test_size=0.2, stratify=[r["label"] for r in records],
        random_state=split_seed,
    )
    counts = np.bincount([r["label"] for r in development], minlength=5)
    if counts.min() < folds:
        raise ValueError(f"Each development class needs at least {folds} trials: {counts}")
    # 정렬해 두면 반복 실행에서도 fold 순서가 동일하게 재현된다.
    development = sorted(development, key=lambda r: r["path"])
    test = sorted(test, key=lambda r: r["path"])
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=split_seed)
    fold_records = []
    for train_indices, val_indices in splitter.split(
        development, [r["label"] for r in development]
    ):
        fold_records.append({
            "train": [development[i] for i in train_indices],
            "validation": [development[i] for i in val_indices],
        })
    manifest = {
        "schema_version": 2, "data_root": str(root), "split_seed": split_seed,
        "subjects": SUBJECTS, "test": test, "development": development,
        "folds": fold_records,
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    def paths(records):
        names = [r["path"] for r in records]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate trial in split")
        if {r["label"] for r in records} != set(range(5)):
            raise ValueError("Each split must contain all five subjects")
        return set(names)

    dev, test = paths(manifest["development"]), paths(manifest["test"])
    if dev & test:
        raise ValueError("Development/test trial overlap")
    val_all = []
    for fold in manifest["folds"]:
        train, val = paths(fold["train"]), paths(fold["validation"])
        if train & val or train | val != dev:
            raise ValueError("Invalid train/validation partition")
        val_all.extend(val)
    if len(val_all) != len(dev) or set(val_all) != dev:
        raise ValueError("Each development trial must validate exactly once")


def read_trial(record, root):
    root = Path(root).resolve()
    path = (root / record["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Trial path escapes data directory")
    if file_digest(path) != record["sha256"]:
        raise ValueError(f"CSV changed since split creation: {path}")
    signal = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if signal.shape[1] != 2 or len(signal) < WINDOW or not np.isfinite(signal).all():
        raise ValueError(f"Expected finite (time >= {WINDOW}, 2) CSV: {path}, {signal.shape}")
    return signal


def transform_trial(signal, refilter=False, hop=EVAL_HOP, cwt_norm="none"):
    """(시간, 2) -> ((윈도우수, 3, 32, 300) float16, (윈도우수,) 시작 샘플 인덱스)."""
    if hop <= 0 or hop > WINDOW:
        raise ValueError(f"hop must be in 1..{WINDOW}")
    if cwt_norm not in ("none", "log_minmax"):
        raise ValueError("cwt_norm must be 'none' or 'log_minmax'")
    if refilter:
        # 배포 CSV는 이미 필터링되어 있다. 재현용 legacy 옵션이다.
        b, a = iirnotch(60, 30, fs=FS)
        signal = filtfilt(b, a, signal, axis=0)
        b, a = butter(4, [20, 499], btype="bandpass", fs=FS)
        signal = filtfilt(b, a, signal, axis=0)
    windows, starts = [], []
    for start in range(0, len(signal) - WINDOW + 1, hop):
        window = signal[start:start + WINDOW]
        window = (window - window.min()) / (window.max() - window.min() + 1e-8)
        maps = [np.abs(pywt.cwt(window[:, ch], SCALES, WAVELET)[0]) for ch in range(2)]
        maps.append((maps[0] + maps[1]) / 2)
        tensor = np.stack(maps)
        if cwt_norm == "log_minmax":
            # |CWT|는 스케일축을 따라 크기 차가 수십 배다. 압축 후 [0,1]로 맞춘다.
            tensor = np.log1p(tensor)
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
        windows.append(tensor)
        starts.append(start)
    result = np.asarray(windows, dtype=np.float32)
    if not len(result) or not np.isfinite(result).all():
        raise ValueError("No finite CWT windows generated")
    return result.astype(np.float16), np.asarray(starts, dtype=np.int64)


def build_dataset(records, root, refilter=False, hop=EVAL_HOP, cwt_norm="none", cache=True):
    """반환 TensorDataset: (입력 float16, 라벨, 시행 id, 윈도우 시작 샘플)."""
    xs, ys, ids, starts = [], [], [], []
    for trial_id, record in enumerate(records):
        key = (record["sha256"], refilter, hop, cwt_norm)
        if cache and key in _CACHE:
            windows, offsets = _CACHE[key]
        else:
            windows, offsets = transform_trial(
                read_trial(record, root), refilter=refilter, hop=hop, cwt_norm=cwt_norm
            )
            if cache:
                _CACHE[key] = (windows, offsets)
        xs.append(windows)
        ys.extend([record["label"]] * len(windows))
        ids.extend([trial_id] * len(windows))
        starts.append(offsets)
        if (trial_id + 1) % 25 == 0 or trial_id + 1 == len(records):
            print(f"CWT trials {trial_id + 1}/{len(records)}", flush=True)
    if not xs:
        raise ValueError("Empty dataset")
    return TensorDataset(
        torch.from_numpy(np.concatenate(xs)),
        torch.tensor(ys, dtype=torch.long),
        torch.tensor(ids, dtype=torch.long),
        torch.from_numpy(np.concatenate(starts)),
    )
