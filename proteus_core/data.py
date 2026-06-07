import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def length_align(X, seq_len):
    if seq_len < X.shape[-1]:
        X = X[..., :seq_len]
    if seq_len > X.shape[-1]:
        padding_num = seq_len - X.shape[-1]
        pad_width = [(0, 0) for _ in range(len(X.shape) - 1)] + [(0, padding_num)]
        X = np.pad(X, pad_width=pad_width, mode="constant", constant_values=0)
    return X


def apply_feature(X, feature, seq_len):
    feature = feature.upper()
    if feature == "DIR":
        X = np.sign(X)
        X = length_align(X, seq_len)
        return torch.tensor(X[:, np.newaxis], dtype=torch.float32)
    if feature == "DT":
        X = length_align(X, seq_len)
        return torch.tensor(X[:, np.newaxis], dtype=torch.float32)
    if feature == "DT2":
        X_dir = np.sign(X)
        X_time = np.abs(X)
        X_time = np.diff(X_time)
        X_time[X_time < 0] = 0
        X_dir = length_align(X_dir, seq_len)[:, np.newaxis]
        X_time = length_align(X_time, seq_len)[:, np.newaxis]
        return torch.tensor(np.concatenate([X_dir, X_time], axis=1), dtype=torch.float32)
    if feature == "ORIGIN":
        X = length_align(X, seq_len)
        return torch.tensor(X, dtype=torch.float32)
    raise ValueError("Unsupported feature '{}'. Core baseline supports DIR, DT, DT2, Origin.".format(feature))


def _load_npz(path):
    data = np.load(path)
    if "X" not in data or "y" not in data:
        raise KeyError("{} must contain X and y arrays.".format(path))
    return data["X"], data["y"]


def _load_json_like(path, view, seq_len, label_to_id=None, label_key=None, sequence_key=None):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import custom_dataset.json_to_npz as converter
    converter.np = np
    build_arrays = converter.build_arrays
    load_records = converter.load_records
    make_label_mapping = converter.make_label_mapping

    records = load_records(path)
    if label_to_id is None:
        label_to_id = make_label_mapping(records, records, label_key)
    X, y, skipped = build_arrays(records, label_to_id, view, seq_len, label_key, sequence_key)
    if skipped:
        print("[Warn] skipped {} unusable records from {}".format(skipped, path))
    return X, y, label_to_id


def load_traffic_split(path, feature="DIR", seq_len=5000, view="dir",
                       label_to_id=None, label_key=None, sequence_key=None):
    """Load a traffic split from npz/json/jsonl/directory and return tensors.

    For a faithful DF+Proteus baseline, store signed packet-size sequences and
    use feature="DIR". The feature transform will apply np.sign(X).
    """
    path = os.path.abspath(path)
    if os.path.isfile(path) and path.lower().endswith(".npz"):
        X, y = _load_npz(path)
    else:
        X, y, label_to_id = _load_json_like(path, view, seq_len, label_to_id, label_key, sequence_key)
    X_tensor = apply_feature(np.asarray(X), feature, seq_len)
    y_tensor = torch.tensor(np.asarray(y), dtype=torch.int64)
    return X_tensor, y_tensor, label_to_id


def make_loader(X, y, batch_size, shuffle, num_workers=0, drop_last=False):
    dataset = TensorDataset(X, y)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )
