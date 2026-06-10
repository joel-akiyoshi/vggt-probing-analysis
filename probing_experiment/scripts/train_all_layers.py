from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
SRC_ROOT = EXPERIMENT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from probing_experiment.data_loading.dataset import validate_probe_artifacts


def default_artifact_root() -> Path:
    data_root = REPO_ROOT / "data"
    if data_root.exists():
        return data_root
    return EXPERIMENT_ROOT / "outputs/probe_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one MLP probe for every layer in a probe dataset."
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=default_artifact_root(),
        help=(
            "Directory containing manifest.csv, targets.npz, and X_train/X_test "
            "layer arrays. Defaults to data/ if present, otherwise "
            "probing_experiment/outputs/probe_dataset."
        ),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=EXPERIMENT_ROOT / "outputs/trained_probes",
        help="Directory where layer_n training outputs are written.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
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
        "--point-coordinate-space",
        choices=["original", "vggt", "vggt_normalized"],
        default="vggt",
        help="Coordinate frame for correspondence targets.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the training commands without running them.",
    )
    return parser.parse_args()


def discover_layers(artifact_root: Path) -> list[int]:
    manifest_path = artifact_root / "manifest.csv"
    with manifest_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    splits_by_layer: dict[int, set[str]] = {}
    for row in rows:
        splits_by_layer.setdefault(int(row["layer_idx"]), set()).add(row["split"])

    layers = [
        layer_idx
        for layer_idx, splits in splits_by_layer.items()
        if {"train", "test"}.issubset(splits)
    ]
    if not layers:
        raise ValueError(f"No layers with both train and test splits in {manifest_path}")
    return sorted(layers)


def train_command(args: argparse.Namespace, layer_idx: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "probing_experiment.probing.train",
        "--artifact-root",
        str(args.artifact_root),
        "--out-dir",
        str(args.out_root / f"layer_{layer_idx}"),
        "--layer-idx",
        str(layer_idx),
        "--point-coordinate-space",
        args.point_coordinate_space,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--scheduler-step-size",
        str(args.scheduler_step_size),
        "--scheduler-gamma",
        str(args.scheduler_gamma),
        "--hidden-dim",
        str(args.hidden_dim),
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
    ]


def command_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [str(SRC_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    return env


def main() -> None:
    args = parse_args()
    args.artifact_root = args.artifact_root.resolve()
    args.out_root = args.out_root.resolve()

    summary = validate_probe_artifacts(args.artifact_root)
    layers = discover_layers(args.artifact_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    print(
        "Validated probe artifacts: "
        f"{summary['manifest_rows']} manifest rows, "
        f"{summary['pair_targets']} pair targets, "
        f"{summary['x_arrays']} X arrays."
    )
    print(f"Training layers: {layers}")
    print(f"Output root: {args.out_root}")

    env = command_env()
    for layer_idx in layers:
        command = train_command(args, layer_idx)
        print()
        print(f"Layer {layer_idx}: {' '.join(command)}")
        if not args.dry_run:
            subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
