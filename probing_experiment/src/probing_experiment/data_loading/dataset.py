from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class ProbePairDataset(Dataset):
    """
    Artifact-backed dataset for VGGT camera-token probing.

    Each item corresponds to one ETH3D image pair at one cached VGGT aggregator
    layer. X rows are fixed-size camera-token features; targets are variable
    length correspondence arrays loaded from a shared NPZ sidecar.
    """

    def __init__(
        self,
        artifact_root: str | Path,
        split: str,
        layer_idx: int,
        *,
        manifest_name: str = "manifest.csv",
        targets_name: str = "targets.npz",
        mmap_x: bool = True,
    ) -> None:
        self.artifact_root = Path(artifact_root)
        self.split = split
        self.layer_idx = int(layer_idx)

        manifest_path = self.artifact_root / manifest_name
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with manifest_path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))

        self.rows: list[dict[str, str]] = [
            row
            for row in rows
            if row["split"] == self.split and int(row["layer_idx"]) == self.layer_idx
        ]
        if not self.rows:
            raise ValueError(
                f"No rows found for split={self.split!r}, layer_idx={self.layer_idx}"
            )

        x_path = self.artifact_root / f"X_{self.split}_layer_{self.layer_idx}.npy"
        if not x_path.exists():
            raise FileNotFoundError(f"X array not found: {x_path}")

        mmap_mode = "r" if mmap_x else None
        self.X = np.load(x_path, mmap_mode=mmap_mode)
        self.targets = np.load(self.artifact_root / targets_name)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        x_row = int(row["x_row"])
        pair_key = pair_target_key(
            row["scene"],
            int(row["image_id1"]),
            int(row["image_id2"]),
        )

        x = np.array(self.X[x_row], dtype=np.float32, copy=True)
        pts1 = np.array(self.targets[f"{pair_key}__pts1"], dtype=np.float32, copy=True)
        pts2 = np.array(self.targets[f"{pair_key}__pts2"], dtype=np.float32, copy=True)
        point3d_ids = np.array(
            self.targets[f"{pair_key}__point3d_ids"],
            dtype=np.int64,
            copy=True,
        )

        return {
            "x": torch.from_numpy(x),
            "pts1": torch.from_numpy(pts1),
            "pts2": torch.from_numpy(pts2),
            "point3d_ids": torch.from_numpy(point3d_ids),
            "metadata": {
                "sample_id": row["sample_id"],
                "scene": row["scene"],
                "layer_idx": int(row["layer_idx"]),
                "image_id1": int(row["image_id1"]),
                "image_id2": int(row["image_id2"]),
                "image_path1": row["image_path1"],
                "image_path2": row["image_path2"],
                "num_matches": int(row["num_matches"]),
                "x_row": x_row,
            },
        }


def pair_target_key(scene: str, image_id1: int, image_id2: int) -> str:
    """Stable key used for pair-level target arrays in targets.npz."""
    return f"{scene}__{int(image_id1)}__{int(image_id2)}"


def probe_pair_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate probe samples while preserving variable-length correspondence targets.

    X can be stacked because every feature row has shape (4096,). Correspondence
    arrays are returned as per-sample lists because each image pair has a
    different number of matches.
    """
    return {
        "x": torch.stack([item["x"] for item in batch], dim=0),
        "pts1": [item["pts1"] for item in batch],
        "pts2": [item["pts2"] for item in batch],
        "point3d_ids": [item["point3d_ids"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }


def validate_probe_artifacts(
    artifact_root: str | Path,
    *,
    manifest_name: str = "manifest.csv",
    targets_name: str = "targets.npz",
    min_matches: int = 100,
) -> dict[str, int]:
    """
    Validate the probe dataset artifact structure.

    Returns a small count summary if all checks pass. Raises on missing arrays,
    shape mismatches, invalid x_row values, or too few correspondences.
    """
    artifact_root = Path(artifact_root)
    manifest_path = artifact_root / manifest_name
    targets_path = artifact_root / targets_name

    with manifest_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    targets = np.load(targets_path)

    rows_by_split_layer: Counter[tuple[str, int]] = Counter()
    pair_keys: set[str] = set()

    for row in rows:
        split = row["split"]
        layer_idx = int(row["layer_idx"])
        x_row = int(row["x_row"])
        rows_by_split_layer[(split, layer_idx)] += 1

        pair_key = pair_target_key(
            row["scene"],
            int(row["image_id1"]),
            int(row["image_id2"]),
        )
        pair_keys.add(pair_key)

        pts1 = targets[f"{pair_key}__pts1"]
        pts2 = targets[f"{pair_key}__pts2"]
        point3d_ids = targets[f"{pair_key}__point3d_ids"]
        if pts1.ndim != 2 or pts1.shape[1] != 2:
            raise ValueError(f"{pair_key} pts1 has invalid shape {pts1.shape}")
        if pts2.ndim != 2 or pts2.shape[1] != 2:
            raise ValueError(f"{pair_key} pts2 has invalid shape {pts2.shape}")
        if point3d_ids.ndim != 1:
            raise ValueError(
                f"{pair_key} point3d_ids has invalid shape {point3d_ids.shape}"
            )
        if pts1.shape[0] != pts2.shape[0] or pts1.shape[0] != point3d_ids.shape[0]:
            raise ValueError(f"{pair_key} target arrays disagree on match count")
        if pts1.shape[0] < min_matches:
            raise ValueError(f"{pair_key} has only {pts1.shape[0]} matches")

        x_path = artifact_root / f"X_{split}_layer_{layer_idx}.npy"
        x_array = np.load(x_path, mmap_mode="r")
        if x_array.ndim != 2 or x_array.shape[1] != 4096:
            raise ValueError(f"{x_path} has invalid shape {x_array.shape}")
        if x_row < 0 or x_row >= x_array.shape[0]:
            raise ValueError(f"{row['sample_id']} has invalid x_row={x_row}")

    for (split, layer_idx), count in rows_by_split_layer.items():
        x_path = artifact_root / f"X_{split}_layer_{layer_idx}.npy"
        x_array = np.load(x_path, mmap_mode="r")
        if x_array.shape[0] != count:
            raise ValueError(
                f"{x_path} has {x_array.shape[0]} rows, manifest has {count}"
            )

    return {
        "manifest_rows": len(rows),
        "pair_targets": len(pair_keys),
        "x_arrays": len(rows_by_split_layer),
    }
