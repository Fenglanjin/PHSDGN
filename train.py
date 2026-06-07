"""Training script for PHSDGN on IEEE 24-bus inertia-estimation datasets."""

from __future__ import annotations

import argparse
import os
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import InertiaDataset, get_ieee24_adj, prepare_inertia_data
from metrics import evaluate_all
from model import PHSDGN


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes", "y"}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PHSDGN for power-system inertia estimation.")
    parser.add_argument("--data_path", type=str, default="train_3rd_wrv.csv")
    parser.add_argument("--model", type=str, default="PHSDGN")
    parser.add_argument("--F", type=int, default=2)
    parser.add_argument("--num_nodes", type=int, default=24)
    parser.add_argument("--tem_size", type=int, default=200)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--Kt", type=int, default=7)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--decay", type=float, default=0.98)
    parser.add_argument("--out_dir", type=str, default="./output")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--acc_tol", type=float, default=0.05)

    parser.add_argument("--train_ratio", type=float, default=0.80)
    parser.add_argument("--val_ratio", type=float, default=0.20)
    parser.add_argument("--split_seed", type=int, default=2026)
    parser.add_argument("--split_shuffle", type=str2bool, default=False)

    parser.add_argument("--use_dynamic_graph", type=str2bool, default=True)
    parser.add_argument("--use_physics_conv", type=str2bool, default=True)
    parser.add_argument("--use_std_pooling", type=str2bool, default=True)
    parser.add_argument("--log_file", type=str, default=None)
    return parser.parse_args()


def physics_consistency_loss(h_hat: torch.Tensor, stats: dict[str, Any]) -> torch.Tensor:
    """Optional soft range penalty for equivalent inertia predictions."""
    y_mean = torch.tensor(stats["y_mean"], device=h_hat.device)
    y_std = torch.tensor(stats["y_std"], device=h_hat.device)
    h_real = h_hat * y_std + y_mean
    below = torch.relu(3.0 - h_real)
    above = torch.relu(h_real - 8.0)
    return (below.pow(2) + above.pow(2)).mean()


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, tol: float = 0.05) -> dict[str, float]:
    model.eval()
    all_pred, all_true = [], []
    for x, h_true in loader:
        x = x.to(device)
        h_hat, _ = model(x)
        h_hat = h_hat * loader.dataset.y_std + loader.dataset.y_mean
        h_true = h_true * loader.dataset.y_std + loader.dataset.y_mean
        all_pred.append(h_hat.cpu().numpy())
        all_true.append(h_true.numpy())
    return evaluate_all(np.concatenate(all_true), np.concatenate(all_pred), tol=tol)


def build_run_tag(args: argparse.Namespace) -> str:
    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
    return "_".join([
        dataset_name,
        f"DG{int(args.use_dynamic_graph)}",
        f"PE{int(args.use_physics_conv)}",
        f"SP{int(args.use_std_pooling)}",
    ])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    run_tag = build_run_tag(args)
    log_file_path = args.log_file or os.path.join("logs", f"{args.model}_{run_tag}.log")
    ckpt_path = os.path.join(args.out_dir, f"{args.model}_{run_tag}_best.pt")

    with open(log_file_path, "w", encoding="utf-8") as log_file:
        def log_print(*items: object) -> None:
            print(*items)
            print(*items, file=log_file, flush=True)

        log_print(f"=== PHSDGN experiment: {run_tag} ===")
        log_print(f"Dataset: {args.data_path}")
        log_print(f"Device: {device}")
        log_print(
            "Config: "
            f"K={args.K}, Kt={args.Kt}, alpha={args.alpha}, hidden_dim={args.hidden_dim}, "
            f"use_dynamic_graph={args.use_dynamic_graph}, "
            f"use_physics_conv={args.use_physics_conv}, "
            f"use_std_pooling={args.use_std_pooling}"
        )

        seed_everything(args.split_seed)
        static_adj = get_ieee24_adj().to(device)
        log_print(f"Loaded IEEE 24-bus adjacency: shape={tuple(static_adj.shape)}, density={static_adj.mean():.3f}")

        prepared = prepare_inertia_data(
            data_path=args.data_path,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.split_seed,
            shuffle=args.split_shuffle,
            normalize=True,
        )
        stats = prepared["stats"]
        split_indices = prepared["split_indices"]

        train_set = InertiaDataset(
            X=prepared["X"],
            y=prepared["y"],
            indices=split_indices["train"],
            split_name="train",
            pmu_indices=prepared["pmu_indices"],
            normalize=True,
            stats=stats,
        )
        val_set = InertiaDataset(
            X=prepared["X"],
            y=prepared["y"],
            indices=split_indices["val"],
            split_name="val",
            pmu_indices=prepared["pmu_indices"],
            normalize=True,
            stats=stats,
        )
        log_print(
            f"Split sizes: train={len(split_indices['train'])}, "
            f"val={len(split_indices['val'])}, test={len(split_indices['test'])}"
        )

        pin_memory = device.type == "cuda"
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

        model = PHSDGN(
            F=args.F,
            num_nodes=args.num_nodes,
            tem_size=args.tem_size,
            static_adj=static_adj,
            K=args.K,
            Kt=args.Kt,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            alpha=args.alpha,
            use_dynamic_graph=args.use_dynamic_graph,
            use_physics_conv=args.use_physics_conv,
            use_std_pooling=args.use_std_pooling,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log_print(f"Trainable parameters: {n_params:,}")

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay)

        best_val_mse = float("inf")
        best_epoch = 0
        patience_cnt = 0
        epoch_times = []
        start_time = time.time()

        for epoch in range(1, args.max_epoch + 1):
            epoch_start = time.time()
            model.train()
            train_losses = []

            for x, h_true in train_loader:
                x = x.to(device)
                h_true = h_true.to(device)
                h_hat, _ = model(x)
                loss = F.mse_loss(h_hat, h_true)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            scheduler.step()
            val_metrics = evaluate(model, val_loader, device, tol=args.acc_tol)
            val_mse = val_metrics["MSE"]
            if device.type == "cuda":
                torch.cuda.synchronize()
            epoch_time = (time.time() - epoch_start) * 1000.0
            epoch_times.append(epoch_time)

            log_print(
                f"Epoch {epoch:03d} | loss={np.mean(train_losses):.6f} "
                f"| val_MSE={val_mse:.6f} | val_MAPE={val_metrics['MAPE']:.4f}% "
                f"| val_ACC={val_metrics['ACC']:.2f}% | time={epoch_time:.1f} ms"
            )

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_epoch = epoch
                patience_cnt = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "epoch": epoch,
                        "val_mse": val_mse,
                        "stats": stats,
                        "args": vars(args),
                        "split_meta": {
                            "train_ratio": args.train_ratio,
                            "val_ratio": args.val_ratio,
                            "split_seed": args.split_seed,
                            "split_shuffle": args.split_shuffle,
                            "train_indices": split_indices["train"].tolist(),
                            "val_indices": split_indices["val"].tolist(),
                            "test_indices": split_indices["test"].tolist(),
                        },
                    },
                    ckpt_path,
                )
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    log_print(f"Early stopping at epoch {epoch}; best_epoch={best_epoch}.")
                    break

        log_print(f"Average epoch time: {np.mean(epoch_times):.1f} ms")
        log_print(f"Best epoch: {best_epoch}, best val_MSE: {best_val_mse:.6f}")

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        final_metrics = evaluate(model, val_loader, device, tol=args.acc_tol)

        elapsed = time.time() - start_time
        log_print(f"Final validation metrics with tol={args.acc_tol * 100:.1f}%:")
        for key, value in final_metrics.items():
            log_print(f"  {key:10s}: {value:.6f}")
        log_print(f"Elapsed time: {elapsed:.1f} s ({elapsed / 60:.1f} min)")
        log_print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
