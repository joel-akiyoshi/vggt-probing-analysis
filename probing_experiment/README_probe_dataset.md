# VGGT Camera-Token Probe Dataset

This document describes the artifact format for MLP probe training data built
from ETH3D image pairs and VGGT aggregator camera-token activations.

## Goal

Each dataset sample represents one ETH3D image pair at one VGGT aggregator layer:

- `X`: ordered concatenation of the two image camera tokens, shape `(4096,)`
- `y`: COLMAP-derived correspondences for that exact image pair
- metadata: scene, image IDs, image paths, layer index, split, and X row index

The probe predicts pair geometry from `X`; the Sampson loss consumes the stored
correspondences.

## Split

The default split is scene-level:

- Test scenes: `meadow`, `pipes`
- Training scenes: every other scene under `probing_experiment/data/highres_train`
- Minimum correspondences per pair: `100`

Scene-level splitting avoids train/test leakage from nearby views of the same
scene.

## Artifact Layout

The builder writes:

```text
manifest.csv
X_train_layer_0.npy
X_test_layer_0.npy
X_train_layer_1.npy
X_test_layer_1.npy
...
targets.npz
```

`manifest.csv` has one row per `(scene, image_id1, image_id2, layer_idx)` sample:

```text
sample_id,split,scene,layer_idx,image_id1,image_id2,image_path1,image_path2,num_matches,x_row
```

`targets.npz` stores variable-length pair targets keyed by pair:

```text
{scene}__{image_id1}__{image_id2}__pts1
{scene}__{image_id1}__{image_id2}__pts2
{scene}__{image_id1}__{image_id2}__point3d_ids
```

The pair ordering is fixed:

- `image_id1` maps to camera token 0 and `pts1`
- `image_id2` maps to camera token 1 and `pts2`

## Building

Smoke test one or two pairs first:

```bash
python probing_experiment/scripts/build_probe_dataset.py \
  --scenes courtyard \
  --max-pairs-per-scene 2 \
  --out-dir probing_experiment/outputs/probe_dataset_smoke
```

Full build:

```bash
python probing_experiment/scripts/build_probe_dataset.py
```

## Loading

Use `ProbePairDataset` for training or evaluation:

```python
from torch.utils.data import DataLoader
from probing_experiment.data_loading.dataset import ProbePairDataset, probe_pair_collate

dataset = ProbePairDataset(
    "probing_experiment/outputs/probe_dataset",
    split="train",
    layer_idx=0,
)

loader = DataLoader(dataset, batch_size=8, collate_fn=probe_pair_collate)
```

Each item returns:

```python
{
    "x": torch.float32,          # shape (4096,)
    "pts1": torch.float32,       # shape (N, 2)
    "pts2": torch.float32,       # shape (N, 2)
    "point3d_ids": torch.long,   # shape (N,)
    "metadata": {...},
}
```

The collate function stacks `x` and keeps correspondence arrays as lists because
the number of matches varies per image pair.
