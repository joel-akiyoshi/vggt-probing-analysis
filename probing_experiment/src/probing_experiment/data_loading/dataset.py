from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ImagePreprocessTransform:
    """Affine map from original image pixels into VGGT input pixels."""

    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    output_width: int
    output_height: int
    scale_x: float
    scale_y: float
    offset_x: float
    offset_y: float

    def apply(self, points: np.ndarray) -> np.ndarray:
        transformed = points.astype(np.float32, copy=True)
        transformed[:, 0] = transformed[:, 0] * self.scale_x + self.offset_x
        transformed[:, 1] = transformed[:, 1] * self.scale_y + self.offset_y
        return transformed

    def valid_mask(self, points: np.ndarray) -> np.ndarray:
        return (
            (points[:, 0] >= 0.0)
            & (points[:, 0] < float(self.output_width))
            & (points[:, 1] >= 0.0)
            & (points[:, 1] < float(self.output_height))
        )


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
        point_coordinate_space: str = "original",
        preprocess_mode: str = "crop",
        target_size: int = 518,
        min_valid_matches: int = 1,
    ) -> None:
        self.artifact_root = Path(artifact_root)
        self.split = split
        self.layer_idx = int(layer_idx)
        self.point_coordinate_space = point_coordinate_space
        self.preprocess_mode = preprocess_mode
        self.target_size = int(target_size)
        self.min_valid_matches = int(min_valid_matches)
        self._pair_transform_cache: dict[
            tuple[str, str],
            tuple[ImagePreprocessTransform, ImagePreprocessTransform],
        ] = {}

        valid_coordinate_spaces = {"original", "vggt", "vggt_normalized"}
        if self.point_coordinate_space not in valid_coordinate_spaces:
            raise ValueError(
                "point_coordinate_space must be one of "
                f"{sorted(valid_coordinate_spaces)}, got {self.point_coordinate_space!r}"
            )
        if self.preprocess_mode not in {"crop", "pad"}:
            raise ValueError(
                f"preprocess_mode must be 'crop' or 'pad', got {self.preprocess_mode!r}"
            )

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
        transform_metadata: dict[str, Any] | None = None

        if self.point_coordinate_space != "original":
            (
                pts1,
                pts2,
                point3d_ids,
                transform_metadata,
            ) = self._transform_correspondences(
                pts1=pts1,
                pts2=pts2,
                point3d_ids=point3d_ids,
                image_path1=row["image_path1"],
                image_path2=row["image_path2"],
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
                "num_valid_matches": int(len(point3d_ids)),
                "x_row": x_row,
                "point_coordinate_space": self.point_coordinate_space,
                "preprocess_mode": self.preprocess_mode,
                "preprocess": transform_metadata,
            },
        }

    def _transform_correspondences(
        self,
        *,
        pts1: np.ndarray,
        pts2: np.ndarray,
        point3d_ids: np.ndarray,
        image_path1: str,
        image_path2: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        transform1, transform2 = self._pair_preprocess_transforms(
            image_path1,
            image_path2,
        )
        pts1_vggt = transform1.apply(pts1)
        pts2_vggt = transform2.apply(pts2)

        valid = transform1.valid_mask(pts1_vggt) & transform2.valid_mask(pts2_vggt)
        if int(valid.sum()) < self.min_valid_matches:
            raise ValueError(
                "Too few valid correspondences after VGGT preprocessing transform: "
                f"{int(valid.sum())} < {self.min_valid_matches} for "
                f"{image_path1}, {image_path2}"
            )

        pts1_out = pts1_vggt[valid].astype(np.float32, copy=False)
        pts2_out = pts2_vggt[valid].astype(np.float32, copy=False)
        point3d_ids_out = point3d_ids[valid].astype(np.int64, copy=False)

        if self.point_coordinate_space == "vggt_normalized":
            pts1_out = _normalize_vggt_points(pts1_out, transform1)
            pts2_out = _normalize_vggt_points(pts2_out, transform2)

        metadata = {
            "original_matches": int(len(point3d_ids)),
            "valid_matches": int(len(point3d_ids_out)),
            "dropped_matches": int(len(point3d_ids) - len(point3d_ids_out)),
            "image1": _transform_metadata(transform1),
            "image2": _transform_metadata(transform2),
        }
        return pts1_out, pts2_out, point3d_ids_out, metadata

    def _pair_preprocess_transforms(
        self, image_path1: str, image_path2: str
    ) -> tuple[ImagePreprocessTransform, ImagePreprocessTransform]:
        cache_key = (image_path1, image_path2)
        if cache_key not in self._pair_transform_cache:
            self._pair_transform_cache[cache_key] = compute_pair_preprocess_transforms(
                image_path1,
                image_path2,
                mode=self.preprocess_mode,
                target_size=self.target_size,
            )
        return self._pair_transform_cache[cache_key]


def pair_target_key(scene: str, image_id1: int, image_id2: int) -> str:
    """Stable key used for pair-level target arrays in targets.npz."""
    return f"{scene}__{int(image_id1)}__{int(image_id2)}"


def compute_pair_preprocess_transforms(
    image_path1: str | Path,
    image_path2: str | Path,
    *,
    mode: str = "crop",
    target_size: int = 518,
) -> tuple[ImagePreprocessTransform, ImagePreprocessTransform]:
    """
    Match vggt.utils.load_fn.load_and_preprocess_images coordinate transforms.

    The returned transforms map original image pixel coordinates into the final
    tensor pixel grid that VGGT receives, including resize, crop/pad, and the
    extra pair-level padding applied when the two processed images differ in
    shape.
    """
    if mode not in {"crop", "pad"}:
        raise ValueError(f"mode must be 'crop' or 'pad', got {mode!r}")

    base1 = _single_preprocess_transform(
        image_path1,
        mode=mode,
        target_size=target_size,
    )
    base2 = _single_preprocess_transform(
        image_path2,
        mode=mode,
        target_size=target_size,
    )

    max_height = max(base1.output_height, base2.output_height)
    max_width = max(base1.output_width, base2.output_width)
    return (
        _add_pair_padding(base1, max_width=max_width, max_height=max_height),
        _add_pair_padding(base2, max_width=max_width, max_height=max_height),
    )


def _single_preprocess_transform(
    image_path: str | Path,
    *,
    mode: str,
    target_size: int,
) -> ImagePreprocessTransform:
    with Image.open(image_path) as img:
        original_width, original_height = img.size

    if mode == "pad":
        if original_width >= original_height:
            resized_width = target_size
            resized_height = (
                round(original_height * (resized_width / original_width) / 14) * 14
            )
        else:
            resized_height = target_size
            resized_width = (
                round(original_width * (resized_height / original_height) / 14) * 14
            )

        pad_left = max(0, target_size - resized_width) // 2
        pad_top = max(0, target_size - resized_height) // 2
        output_width = target_size
        output_height = target_size
        offset_x = float(pad_left)
        offset_y = float(pad_top)
    else:
        resized_width = target_size
        resized_height = (
            round(original_height * (resized_width / original_width) / 14) * 14
        )
        crop_top = max(0, (resized_height - target_size) // 2)
        output_width = resized_width
        output_height = min(resized_height, target_size)
        offset_x = 0.0
        offset_y = -float(crop_top)

    return ImagePreprocessTransform(
        original_width=original_width,
        original_height=original_height,
        resized_width=resized_width,
        resized_height=resized_height,
        output_width=output_width,
        output_height=output_height,
        scale_x=float(resized_width) / float(original_width),
        scale_y=float(resized_height) / float(original_height),
        offset_x=offset_x,
        offset_y=offset_y,
    )


def _add_pair_padding(
    transform: ImagePreprocessTransform,
    *,
    max_width: int,
    max_height: int,
) -> ImagePreprocessTransform:
    pad_left = max(0, max_width - transform.output_width) // 2
    pad_top = max(0, max_height - transform.output_height) // 2
    if pad_left == 0 and pad_top == 0:
        return transform

    return ImagePreprocessTransform(
        original_width=transform.original_width,
        original_height=transform.original_height,
        resized_width=transform.resized_width,
        resized_height=transform.resized_height,
        output_width=max_width,
        output_height=max_height,
        scale_x=transform.scale_x,
        scale_y=transform.scale_y,
        offset_x=transform.offset_x + float(pad_left),
        offset_y=transform.offset_y + float(pad_top),
    )


def _normalize_vggt_points(
    points: np.ndarray,
    transform: ImagePreprocessTransform,
) -> np.ndarray:
    normalized = points.astype(np.float32, copy=True)
    normalized[:, 0] = normalized[:, 0] / float(transform.output_width - 1)
    normalized[:, 1] = normalized[:, 1] / float(transform.output_height - 1)
    return normalized


def _transform_metadata(transform: ImagePreprocessTransform) -> dict[str, Any]:
    return {
        "original_size": (transform.original_width, transform.original_height),
        "resized_size": (transform.resized_width, transform.resized_height),
        "output_size": (transform.output_width, transform.output_height),
        "scale": (transform.scale_x, transform.scale_y),
        "offset": (transform.offset_x, transform.offset_y),
    }


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
