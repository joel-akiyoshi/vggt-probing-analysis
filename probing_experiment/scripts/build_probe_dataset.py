from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from vggt.utils.load_fn import load_and_preprocess_images

from probing_experiment.data_loading.colmap_images import load_images_txt
from probing_experiment.data_loading.correspondences import (
    get_all_image_pair_correspondences,
)
from probing_experiment.data_loading.dataset import pair_target_key
from probing_experiment.data_loading.dataset import validate_probe_artifacts
from probing_experiment.vggt_utils.inference import run_vggt_with_camera_token_cache
from probing_experiment.vggt_utils.model import load_vggt


DEFAULT_HOLDOUT_SCENES = ("meadow", "pipes")
MANIFEST_COLUMNS = (
    "sample_id",
    "split",
    "scene",
    "layer_idx",
    "image_id1",
    "image_id2",
    "image_path1",
    "image_path2",
    "num_matches",
    "x_row",
)


def parse_args() -> argparse.Namespace:
    experiment_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build VGGT camera-token probe artifacts from ETH3D scenes."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=experiment_root / "data/highres_train",
        help="Directory containing ETH3D scene folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=experiment_root / "outputs/probe_dataset",
        help="Directory where manifest, X arrays, and targets are written.",
    )
    parser.add_argument(
        "--holdout-scenes",
        nargs="+",
        default=list(DEFAULT_HOLDOUT_SCENES),
        help="Scene names assigned to the test split.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Optional subset of scene names to build.",
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=100,
        help="Minimum shared COLMAP points required for an image pair.",
    )
    parser.add_argument(
        "--max-pairs-per-scene",
        type=int,
        default=None,
        help="Optional cap for smoke tests. Full build uses all qualifying pairs.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for VGGT inference.",
    )
    return parser.parse_args()


def scene_names(data_root: Path, requested: list[str] | None) -> list[str]:
    available = sorted(path.name for path in data_root.iterdir() if path.is_dir())
    if requested is None:
        return available

    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"Requested scenes not found under {data_root}: {missing}")
    return [name for name in available if name in set(requested)]


def camera_tokens_to_feature(tokens: np.ndarray) -> np.ndarray:
    """
    Convert one layer's camera-token activation to an ordered pair feature.

    Expected input shape is (1, 2, 2048). Output shape is (4096,).
    """
    if tokens.shape != (1, 2, 2048):
        raise ValueError(f"Expected camera tokens with shape (1, 2, 2048), got {tokens.shape}")
    return np.concatenate([tokens[0, 0], tokens[0, 1]], axis=0).astype(np.float32)


def write_manifest(out_dir: Path, rows: list[dict[str, object]]) -> None:
    with (out_dir / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def build_probe_dataset(
    data_root: Path,
    out_dir: Path,
    holdout_scenes: set[str],
    selected_scenes: list[str],
    min_matches: int,
    max_pairs_per_scene: int | None,
    device: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_vggt(device)
    manifest_rows: list[dict[str, object]] = []
    targets: dict[str, np.ndarray] = {}
    x_rows_by_split_layer: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)

    for scene in selected_scenes:
        split = "test" if scene in holdout_scenes else "train"
        scene_dir = data_root / scene
        calibration_dir = scene_dir / "dslr_calibration_undistorted"
        image_dir = scene_dir / "images"

        print(f"\nScene {scene} ({split})")
        image_records = load_images_txt(calibration_dir / "images.txt")
        pair_corrs = get_all_image_pair_correspondences(
            image_records,
            min_matches=min_matches,
        )
        pair_items = list(pair_corrs.items())
        if max_pairs_per_scene is not None:
            pair_items = pair_items[:max_pairs_per_scene]
        print(f"  qualifying pairs: {len(pair_items)}")

        for pair_index, ((image_id1, image_id2), (pts1, pts2, point3d_ids)) in enumerate(
            pair_items,
            start=1,
        ):
            im1_path = image_dir / image_records[image_id1].name
            im2_path = image_dir / image_records[image_id2].name
            print(
                f"  pair {pair_index}/{len(pair_items)}: "
                f"{image_id1}, {image_id2} ({len(point3d_ids)} matches)"
            )

            pair_key = pair_target_key(scene, image_id1, image_id2)
            targets[f"{pair_key}__pts1"] = pts1.astype(np.float32)
            targets[f"{pair_key}__pts2"] = pts2.astype(np.float32)
            targets[f"{pair_key}__point3d_ids"] = point3d_ids.astype(np.int64)

            images = load_and_preprocess_images([im1_path, im2_path]).to(device)
            predictions, activation_cache = run_vggt_with_camera_token_cache(
                model,
                images,
                device,
            )

            camera_tokens = activation_cache["camera_tokens"]  # dict {layer_idx (int): camera_token (shape 1, 2, 2048)}
            for layer_idx in sorted(camera_tokens):
                tokens = camera_tokens[layer_idx].numpy()
                feature = camera_tokens_to_feature(tokens)  # concat
                x_key = (split, int(layer_idx))
                x_row = len(x_rows_by_split_layer[x_key])
                x_rows_by_split_layer[x_key].append(feature)

                manifest_rows.append(
                    {
                        "sample_id": f"{pair_key}__layer_{layer_idx}",
                        "split": split,
                        "scene": scene,
                        "layer_idx": int(layer_idx),
                        "image_id1": int(image_id1),
                        "image_id2": int(image_id2),
                        "image_path1": str(im1_path),
                        "image_path2": str(im2_path),
                        "num_matches": int(len(point3d_ids)),
                        "x_row": int(x_row),
                    }
                )
            del predictions, images
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    for (split, layer_idx), rows in sorted(x_rows_by_split_layer.items()):
        array = np.stack(rows, axis=0).astype(np.float32)
        np.save(out_dir / f"X_{split}_layer_{layer_idx}.npy", array)
        print(f"Wrote X_{split}_layer_{layer_idx}.npy with shape {array.shape}")

    np.savez(out_dir / "targets.npz", **targets)
    write_manifest(out_dir, manifest_rows)
    summary = validate_probe_artifacts(out_dir, min_matches=min_matches)
    print(f"Wrote manifest with {len(manifest_rows)} rows")
    print(f"Wrote {len(targets) // 3} pair targets")
    print(f"Validation summary: {summary}")


def main() -> None:
    args = parse_args()
    selected_scenes = scene_names(args.data_root, args.scenes)
    build_probe_dataset(
        data_root=args.data_root,
        out_dir=args.out_dir,
        holdout_scenes=set(args.holdout_scenes),
        selected_scenes=selected_scenes,
        min_matches=args.min_matches,
        max_pairs_per_scene=args.max_pairs_per_scene,
        device=args.device,
    )


if __name__ == "__main__":
    main()
