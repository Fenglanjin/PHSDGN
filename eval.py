"""Evaluate a trained PHSDGN checkpoint."""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import InertiaDataset, get_ieee24_adj, prepare_inertia_data
from metrics import evaluate_all
from model import PHSDGN
from train import build_run_tag, str2bool


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PHSDGN checkpoint.")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default="train_3rd_wrv_SNR45db.csv")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--acc_tol", type=float, default=0.05)
    parser.add_argument("--eval_split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--out_dir", type=str, default="./output")
    parser.add_argument("--model", type=str, default="PHSDGN")
    parser.add_argument("--F", type=int, default=2)
    parser.add_argument("--num_nodes", type=int, default=24)
    parser.add_argument("--tem_size", type=int, default=200)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--Kt", type=int, default=7)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.20)
    parser.add_argument("--split_seed", type=int, default=2026)
    parser.add_argument("--split_shuffle", type=str2bool, default=False)
    parser.add_argument("--use_dynamic_graph", type=str2bool, default=True)
    parser.add_argument("--use_physics_conv", type=str2bool, default=True)
    parser.add_argument("--use_std_pooling", type=str2bool, default=True)
    return parser.parse_args()


def checkpoint_args(checkpoint: dict[str, Any], fallback: argparse.Namespace) -> dict[str, Any]:
    """Return training args saved in a checkpoint, with CLI fallback for legacy checkpoints."""
    saved = checkpoint.get("args")
    if isinstance(saved, argparse.Namespace):
        saved = vars(saved)
    if not isinstance(saved, dict):
        saved = vars(fallback)
    return saved


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device, tol: float) -> dict[str, float]:
    model.eval()
    all_pred, all_true = [], []
    y_mean = loader.dataset.y_mean
    y_std = loader.dataset.y_std
    for x, h_true in loader:
        x = x.to(device)
        h_hat, _ = model(x)
        h_hat = h_hat * y_std + y_mean
        h_true = h_true * y_std + y_mean
        all_pred.append(h_hat.cpu().numpy())
        all_true.append(h_true.numpy())
    return evaluate_all(np.concatenate(all_true), np.concatenate(all_pred), tol=tol)


def main() -> None:
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.ckpt_path is None:
        args.ckpt_path = os.path.join(args.out_dir, f"{args.model}_{build_run_tag(args)}_best.pt")

    print(f"Loading checkpoint: {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    train_args = checkpoint_args(checkpoint, args)
    stats = checkpoint.get("stats")
    if stats is None:
        raise ValueError("The checkpoint does not contain train-set normalization statistics.")

    split_meta = checkpoint.get("split_meta", {})
    train_ratio = split_meta.get("train_ratio", train_args.get("train_ratio", args.train_ratio))
    val_ratio = split_meta.get("val_ratio", train_args.get("val_ratio", args.val_ratio))
    split_seed = split_meta.get("split_seed", train_args.get("split_seed", args.split_seed))
    split_shuffle = split_meta.get("split_shuffle", train_args.get("split_shuffle", args.split_shuffle))

    prepared = prepare_inertia_data(
        data_path=args.data_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=split_seed,
        shuffle=split_shuffle,
        normalize=True,
    )
    split_indices = prepared["split_indices"]
    for split_name in ("train", "val", "test"):
        key = f"{split_name}_indices"
        if key in split_meta:
            split_indices[split_name] = np.array(split_meta[key], dtype=np.int64)

    eval_indices = split_indices[args.eval_split]
    if len(eval_indices) == 0:
        raise ValueError(f"eval_split={args.eval_split} is empty for the current split configuration.")

    eval_set = InertiaDataset(
        X=prepared["X"],
        y=prepared["y"],
        indices=eval_indices,
        split_name=args.eval_split,
        pmu_indices=prepared["pmu_indices"],
        normalize=True,
        stats=stats,
    )
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    static_adj = get_ieee24_adj().to(device)
    model = PHSDGN(
        F=train_args.get("F", args.F),
        num_nodes=train_args.get("num_nodes", args.num_nodes),
        tem_size=train_args.get("tem_size", args.tem_size),
        static_adj=static_adj,
        K=train_args.get("K", args.K),
        Kt=train_args.get("Kt", args.Kt),
        hidden_dim=train_args.get("hidden_dim", args.hidden_dim),
        n_layers=train_args.get("n_layers", args.n_layers),
        alpha=train_args.get("alpha", args.alpha),
        use_dynamic_graph=train_args.get("use_dynamic_graph", args.use_dynamic_graph),
        use_physics_conv=train_args.get("use_physics_conv", args.use_physics_conv),
        use_std_pooling=train_args.get("use_std_pooling", args.use_std_pooling),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    metrics = evaluate_model(model, eval_loader, device, tol=args.acc_tol)
    print(f"Evaluation split: {args.eval_split} ({len(eval_set)} samples)")
    print(f"ACC tolerance: {args.acc_tol * 100:.1f}%")
    for key, value in metrics.items():
        print(f"{key:10s}: {value:.6f}")


if __name__ == "__main__":
    main()
