from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from probing_experiment.data_loading.dataset import (
    ProbePairDataset,
    probe_pair_collate,
)
from probing_experiment.probing.loss import sampson_error
from probing_experiment.probing.mlp import FundamentalMatrixMLP


@dataclass(frozen=True)
class TrainConfig:
    artifact_root: str
    out_dir: str
    layer_idx: int = 0
    point_coordinate_space: str = "vggt"
    batch_size: int = 32
    epochs: int = 30
    lr: float = 1e-4
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.5
    hidden_dim: int = 512
    input_dim: int = 4096
    seed: int = 0
    num_workers: int = 0
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a layer-specific MLP probe with Sampson loss."
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("probing_experiment/outputs/probe_dataset"),
        help="Directory containing manifest.csv, targets.npz, and X arrays.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("probing_experiment/outputs/mlp_training/layer_0"),
        help="Directory for metrics.csv, config.json, and best.pt.",
    )
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument(
        "--point-coordinate-space",
        choices=["original", "vggt", "vggt_normalized"],
        default="vggt",
        help="Coordinate frame for correspondence targets.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler-step-size", type=int, default=10)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="Use 'auto', 'cpu', 'cuda', or another torch device string.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Optional cap for smoke tests.",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="Optional cap for smoke tests.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        artifact_root=str(args.artifact_root),
        out_dir=str(args.out_dir),
        layer_idx=args.layer_idx,
        point_coordinate_space=args.point_coordinate_space,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
    )


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataloaders(config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    dataset_kwargs = {
        "artifact_root": config.artifact_root,
        "layer_idx": config.layer_idx,
        "point_coordinate_space": config.point_coordinate_space,
    }
    train_dataset = ProbePairDataset(split="train", **dataset_kwargs)
    test_dataset = ProbePairDataset(split="test", **dataset_kwargs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=probe_pair_collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=probe_pair_collate,
    )
    return train_loader, test_loader


def batch_sampson_loss(
    fundamental: torch.Tensor,
    pts1: list[torch.Tensor],
    pts2: list[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Average per-pair Sampson losses so each image pair has equal weight."""
    losses = []
    for idx, f_i in enumerate(fundamental):
        pts1_i = pts1[idx].to(device, non_blocking=True)
        pts2_i = pts2[idx].to(device, non_blocking=True)
        losses.append(sampson_error(f_i, pts1_i, pts2_i).mean())
    return torch.stack(losses).mean()


def singular_value_metrics(fundamental: torch.Tensor) -> dict[str, float]:
    singular_values = torch.linalg.svdvals(fundamental.detach())
    sigma1 = singular_values[:, 0].clamp_min(1e-12)
    sigma3 = singular_values[:, 2]
    return {
        "sigma3_mean": float(sigma3.mean().item()),
        "sigma3_over_sigma1_mean": float((sigma3 / sigma1).mean().item()),
    }


def run_epoch(
    model: FundamentalMatrixMLP,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_pairs = 0
    sigma3_sum = 0.0
    sigma3_over_sigma1_sum = 0.0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = batch["x"].to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            fundamental = model(x)
            loss = batch_sampson_loss(
                fundamental,
                batch["pts1"],
                batch["pts2"],
                device,
            )
            if is_train:
                loss.backward()
                optimizer.step()

        batch_pairs = int(x.shape[0])
        metrics = singular_value_metrics(fundamental)
        total_loss += float(loss.item()) * batch_pairs
        total_pairs += batch_pairs
        sigma3_sum += metrics["sigma3_mean"] * batch_pairs
        sigma3_over_sigma1_sum += metrics["sigma3_over_sigma1_mean"] * batch_pairs

    if total_pairs == 0:
        raise ValueError("No batches were processed.")

    return {
        "loss": total_loss / total_pairs,
        "pairs": float(total_pairs),
        "sigma3_mean": sigma3_sum / total_pairs,
        "sigma3_over_sigma1_mean": sigma3_over_sigma1_sum / total_pairs,
    }


def write_config(config: TrainConfig, out_dir: Path) -> None:
    with (out_dir / "config.json").open("w") as f:
        json.dump(asdict(config), f, indent=2, sort_keys=True)


def append_metrics(metrics_path: Path, row: dict[str, Any]) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "test_loss",
        "lr",
        "train_pairs",
        "test_pairs",
        "test_sigma3_mean",
        "test_sigma3_over_sigma1_mean",
    ]
    write_header = not metrics_path.exists()
    with metrics_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(
    path: Path,
    *,
    model: FundamentalMatrixMLP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    test_loss: float,
    config: TrainConfig,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "test_loss": test_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": asdict(config),
        },
        path,
    )


def train(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = resolve_device(config.device)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_config(config, out_dir)

    train_loader, test_loader = make_dataloaders(config)
    model = FundamentalMatrixMLP(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.scheduler_step_size,
        gamma=config.scheduler_gamma,
    )

    metrics_path = out_dir / "metrics.csv"
    best_path = out_dir / "best.pt"
    if metrics_path.exists():
        metrics_path.unlink()
    best_test_loss = float("inf")

    print(f"Training layer {config.layer_idx} on {device}")
    print(f"Train pairs: {len(train_loader.dataset)}")
    print(f"Test pairs: {len(test_loader.dataset)}")

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            max_batches=config.max_train_batches,
        )
        with torch.no_grad():
            test_metrics = run_epoch(
                model,
                test_loader,
                device,
                max_batches=config.max_eval_batches,
            )

        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "test_loss": test_metrics["loss"],
            "lr": lr,
            "train_pairs": int(train_metrics["pairs"]),
            "test_pairs": int(test_metrics["pairs"]),
            "test_sigma3_mean": test_metrics["sigma3_mean"],
            "test_sigma3_over_sigma1_mean": test_metrics[
                "sigma3_over_sigma1_mean"
            ],
        }
        append_metrics(metrics_path, row)

        if test_metrics["loss"] < best_test_loss:
            best_test_loss = test_metrics["loss"]
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                test_loss=best_test_loss,
                config=config,
            )

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.6g} "
            f"test_loss={test_metrics['loss']:.6g} "
            f"lr={lr:.3g} "
            f"test_sigma3={test_metrics['sigma3_mean']:.6g} "
            f"test_sigma3/sigma1={test_metrics['sigma3_over_sigma1_mean']:.6g}"
        )
        scheduler.step()

    print(f"Metrics written to {metrics_path}")
    print(f"Best checkpoint written to {best_path}")


def main() -> None:
    train(config_from_args(parse_args()))


if __name__ == "__main__":
    main()
