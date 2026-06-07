"""Data utilities for IEEE 24-bus inertia-estimation experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PMU_INDICES = [0, 1, 12, 15, 17, 20, 21, 22]


def load_raw_inertia_data(data_path: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Load the raw CSV data and build an input tensor with shape [S, F, N, T].

    The source CSV stores PMU measurements in flattened channel blocks. This
    function maps the eight observed PMU buses into a 24-bus tensor and leaves
    unobserved buses as zeros.
    """
    df = pd.read_csv(data_path)
    data_values = df.values

    n_samples = data_values.shape[0]
    num_steps = 200
    num_features = 2
    num_nodes = 24
    x_all = np.zeros((n_samples, num_features, num_nodes, num_steps), dtype=np.float32)

    # Each tuple is (bus_index, feature_0_start, feature_1_start).
    pmu_blocks = [
        (0, 0, 2600),
        (1, 200, 2800),
        (12, 600, 3200),
        (15, 1200, 3800),
        (17, 1400, 4000),
        (20, 1600, 4200),
        (21, 2000, 4600),
        (22, 2400, 5000),
    ]
    for bus, f0_start, f1_start in pmu_blocks:
        x_all[:, 0, bus, :] = data_values[:, f0_start:f0_start + num_steps]
        x_all[:, 1, bus, :] = data_values[:, f1_start:f1_start + num_steps]

    y_all = data_values[:, 7000].astype(np.float32)
    return x_all, y_all, list(PMU_INDICES)


def build_split_indices(
    n_samples: int,
    train_ratio: float = 0.80,
    val_ratio: float = 0.20,
    seed: int = 100,
    shuffle: bool = False,
) -> dict[str, np.ndarray]:
    """Create non-overlapping train/validation/test indices."""
    if train_ratio < 0 or val_ratio < 0:
        raise ValueError("train_ratio and val_ratio must be non-negative.")
    if train_ratio + val_ratio > 1.0:
        raise ValueError("train_ratio + val_ratio must not exceed 1.0.")

    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    if np.intersect1d(train_idx, val_idx).size:
        raise RuntimeError("Train and validation indices overlap.")
    if np.intersect1d(train_idx, test_idx).size:
        raise RuntimeError("Train and test indices overlap.")
    if np.intersect1d(val_idx, test_idx).size:
        raise RuntimeError("Validation and test indices overlap.")
    if train_idx.size + val_idx.size + test_idx.size != n_samples:
        raise RuntimeError("Split index count does not match the sample count.")

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def fit_stats_on_train(
    x_all: np.ndarray,
    y_all: np.ndarray,
    train_idx: np.ndarray,
    pmu_indices: list[int],
) -> dict[str, np.ndarray | np.float32]:
    """Fit normalization statistics only on the training split."""
    x_train = x_all[train_idx][:, :, pmu_indices, :]
    y_train = y_all[train_idx]

    return {
        "mean": x_train.mean(axis=(0, 2, 3), keepdims=True),
        "std": x_train.std(axis=(0, 2, 3), keepdims=True) + 1e-8,
        "y_mean": y_train.mean(),
        "y_std": y_train.std() + 1e-8,
    }


def prepare_inertia_data(
    data_path: str,
    train_ratio: float = 0.80,
    val_ratio: float = 0.20,
    seed: int = 100,
    shuffle: bool = False,
    normalize: bool = True,
) -> dict[str, object]:
    """Load raw data, create data splits, and fit train-only statistics."""
    x_all, y_all, pmu_indices = load_raw_inertia_data(data_path)
    split_indices = build_split_indices(
        n_samples=x_all.shape[0],
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        shuffle=shuffle,
    )
    stats = fit_stats_on_train(x_all, y_all, split_indices["train"], pmu_indices) if normalize else None

    return {
        "X": x_all,
        "y": y_all,
        "pmu_indices": pmu_indices,
        "split_indices": split_indices,
        "stats": stats,
    }


class InertiaDataset(Dataset):
    """Torch dataset for normalized inertia-estimation samples."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
        split_name: str,
        pmu_indices: list[int],
        normalize: bool = False,
        stats: dict[str, object] | None = None,
    ) -> None:
        super().__init__()

        x_split = X[indices].copy()
        y_split = y[indices].copy()

        if normalize:
            if stats is None:
                raise ValueError("stats must be provided when normalize=True.")
            x_split[:, :, pmu_indices, :] = (x_split[:, :, pmu_indices, :] - stats["mean"]) / stats["std"]
            y_split = (y_split - stats["y_mean"]) / stats["y_std"]

        self.X = torch.tensor(x_split, dtype=torch.float32)
        self.y = torch.tensor(y_split, dtype=torch.float32)
        self.mean = stats["mean"] if stats is not None else None
        self.std = stats["std"] if stats is not None else None
        self.y_mean = stats["y_mean"] if stats is not None else None
        self.y_std = stats["y_std"] if stats is not None else None

        print(f"[{split_name.upper()}] loaded: X_shape={tuple(self.X.shape)}, y_shape={tuple(self.y.shape)}")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    def get_stats(self) -> dict[str, object]:
        return {"mean": self.mean, "std": self.std, "y_mean": self.y_mean, "y_std": self.y_std}


def get_ieee24_adj() -> torch.Tensor:
    """Return the undirected physical adjacency matrix of the IEEE 24-bus system."""
    n_nodes = 24
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    edges = [
        (1, 2), (1, 3), (1, 5), (2, 4), (2, 6), (3, 9), (3, 24), (4, 9),
        (5, 10), (6, 10), (7, 8), (8, 9), (8, 10), (9, 11), (9, 12), (10, 11),
        (10, 12), (11, 13), (11, 14), (12, 13), (12, 23), (13, 23), (14, 16),
        (15, 16), (15, 21), (15, 24), (16, 17), (16, 19), (17, 18), (17, 22),
        (18, 21), (19, 20), (20, 23), (21, 22),
    ]
    for u, v in edges:
        adj[u - 1, v - 1] = 1.0
        adj[v - 1, u - 1] = 1.0
    return torch.tensor(adj)
