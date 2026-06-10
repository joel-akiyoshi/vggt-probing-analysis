# MLP Training Data Flow

This note describes the current VGGT camera-token probe artifacts and how they
feed one MLP per VGGT aggregator layer. The intended supervised objective is:

```text
two ETH3D images
  -> VGGT
  -> camera tokens from one aggregator layer
  -> layer-specific MLP
  -> F_pred, shape (3, 3)
  -> Sampson error against COLMAP-derived 2D correspondences
```

## Producer Scripts

### Smoke/debug pair

`probing_experiment/scripts/debug_pair.py` runs VGGT on one ETH3D image pair and
writes low-level debug artifacts:

```text
probing_experiment/outputs/vggt_debug_outputs/
  pair_1_2_camera_tokens.npz
  pair_1_2_vggt_world_points.ply
```

`pair_1_2_camera_tokens.npz` is useful for inspecting raw camera-token captures
before dataset flattening:

```text
cached_layers:   int64,   shape (24,)
camera_layer_0:  float32, shape (1, 2, 2048)
camera_layer_1:  float32, shape (1, 2, 2048)
...
camera_layer_23: float32, shape (1, 2, 2048)
```

The `(1, 2, 2048)` dimensions mean:

```text
batch = 1
sequence/images = 2
camera-token width = 2048
```

For a pair `(image_id1, image_id2)`, token index `0` corresponds to `image_id1`
and token index `1` corresponds to `image_id2`.

`pair_1_2_vggt_world_points.ply` is only a visualization/debug artifact. It is
not part of MLP training.

### Probe dataset build

`probing_experiment/scripts/build_probe_dataset.py` builds the training-ready
artifacts:

```text
probing_experiment/outputs/probe_dataset/
  manifest.csv
  targets.npz
  X_train_layer_0.npy
  X_train_layer_1.npy
  ...
  X_test_layer_0.npy
  X_test_layer_1.npy
  ...
```

The smoke output currently exists at:

```text
probing_experiment/outputs/probe_dataset_smoke/
  manifest.csv
  targets.npz
  X_train_layer_0.npy
  ...
  X_train_layer_23.npy
```

For the smoke run, there are 2 qualifying image pairs from `courtyard`, so every
`X_train_layer_L.npy` has shape:

```text
(2, 4096), float32
```

The full build will have one `X_{split}_layer_{L}.npy` file for each split/layer
combination that has samples. Default holdout scenes are `meadow` and `pipes`,
so those become `test`; all other scenes become `train`.

## X Arrays

Each MLP input row is made by flattening the two camera tokens from one layer:

```python
tokens.shape == (1, 2, 2048)
x = concat(tokens[0, 0], tokens[0, 1])
x.shape == (4096,)
```

File naming:

```text
X_{split}_layer_{layer_idx}.npy
```

Shape:

```text
(num_pairs_in_split, 4096), float32
```

For one layer-specific MLP:

```text
X_batch:     (B, 4096)
MLP_L(X):    (B, 9) or (B, 3, 3)
F_pred:      (B, 3, 3)
```

The current artifact format is designed for one MLP per layer:

```text
MLP_0  trains on X_train_layer_0.npy
MLP_1  trains on X_train_layer_1.npy
...
MLP_23 trains on X_train_layer_23.npy
```

## Manifest

`manifest.csv` has one row per `(image pair, layer)` sample:

```text
sample_id,split,scene,layer_idx,image_id1,image_id2,image_path1,image_path2,num_matches,x_row
```

Example smoke rows:

```text
courtyard__1__2__layer_0,train,courtyard,0,1,2,...,2309,0
courtyard__1__2__layer_1,train,courtyard,1,1,2,...,2309,0
...
courtyard__1__3__layer_0,train,courtyard,0,1,3,...,1788,1
```

Important fields:

```text
split       selects X_train_* or X_test_*
layer_idx   selects the layer-specific X file and MLP
x_row       row index inside X_{split}_layer_{layer_idx}.npy
scene,
image_id1,
image_id2   form the key into targets.npz
num_matches number of 2D correspondences for this pair
```

The manifest is layer-level, but targets are pair-level. All 24 layer rows for
`courtyard__1__2` point to the same correspondence arrays.

## Targets

`targets.npz` stores variable-length correspondence arrays keyed by ordered image
pair:

```text
{scene}__{image_id1}__{image_id2}__pts1
{scene}__{image_id1}__{image_id2}__pts2
{scene}__{image_id1}__{image_id2}__point3d_ids
```

For smoke output:

```text
courtyard__1__2__pts1          float32, shape (2309, 2)
courtyard__1__2__pts2          float32, shape (2309, 2)
courtyard__1__2__point3d_ids   int64,   shape (2309,)
courtyard__1__3__pts1          float32, shape (1788, 2)
courtyard__1__3__pts2          float32, shape (1788, 2)
courtyard__1__3__point3d_ids   int64,   shape (1788,)
```

For a general pair:

```text
pts1:        (N, 2), float32 pixel coordinates in image_id1
pts2:        (N, 2), float32 pixel coordinates in image_id2
point3d_ids: (N,),   int64 COLMAP 3D point IDs shared by both images
```

`pts1[i]` and `pts2[i]` are observations of the same COLMAP 3D point
`point3d_ids[i]`. These are the supervision points for Sampson error.

## Dataset Loader

`ProbePairDataset` in
`probing_experiment/src/probing_experiment/data_loading/dataset.py` binds these
files together:

```python
dataset = ProbePairDataset(
    "probing_experiment/outputs/probe_dataset",
    split="train",
    layer_idx=L,
)
```

Each item returns:

```text
x:            torch.float32, shape (4096,)
pts1:         torch.float32, shape (N, 2)
pts2:         torch.float32, shape (N, 2)
point3d_ids:  torch.int64,   shape (N,)
metadata:     sample_id, scene, layer_idx, image IDs, image paths, num_matches, x_row
```

By default, `pts1` and `pts2` are returned in original ETH3D/COLMAP pixel
coordinates. For Sampson training against an MLP prediction tied to VGGT camera
tokens, prefer loading points in VGGT's preprocessed coordinate frame:

```python
dataset = ProbePairDataset(
    "probing_experiment/outputs/probe_dataset",
    split="train",
    layer_idx=L,
    point_coordinate_space="vggt_normalized",
)
```

Supported point spaces:

```text
original         original ETH3D/COLMAP pixels
vggt             VGGT input tensor pixels after resize/crop/pad
vggt_normalized  VGGT input tensor coordinates normalized to [0, 1]
```

When using `vggt` or `vggt_normalized`, the loader reproduces
`load_and_preprocess_images(..., mode="crop")` for each image path. It transforms
the stored correspondences and filters out matches that fall outside either
image after preprocessing. This can happen because VGGT may crop, and because
COLMAP keypoints can already lie slightly outside image bounds.

The returned metadata includes:

```text
num_matches                  original target count from manifest
num_valid_matches            count after preprocessing-space filtering
metadata["preprocess"]       scale/offset/output-size details for both images
```

`probe_pair_collate` stacks only `x`:

```text
batch["x"]:    (B, 4096)
batch["pts1"]: list of B tensors, each (N_i, 2)
batch["pts2"]: list of B tensors, each (N_i, 2)
```

The correspondence tensors stay as lists because each image pair has a different
number of matches.

## Loss Connection

`probing_experiment/src/probing_experiment/probing/loss.py` already contains:

```python
sampson_error(fundamental, pts1, pts2)
mean_sampson_error(fundamental, pts1, pts2)
```

Expected shapes:

```text
fundamental: (..., 3, 3)
pts1:        (..., N, 2)
pts2:        (..., N, 2)
output:      (..., N)
```

For one sample:

```text
x_i -> MLP_L -> F_pred_i, shape (3, 3)
loss_i = mean(sampson_error(F_pred_i, pts1_i, pts2_i))
```

For a batch with variable `N_i`, the simplest training loop computes per-sample
losses and averages them:

```python
F_pred = model(batch["x"]).reshape(-1, 3, 3)

losses = []
for i, F_i in enumerate(F_pred):
    err_i = sampson_error(F_i, batch["pts1"][i], batch["pts2"][i])
    losses.append(err_i.mean())

loss = torch.stack(losses).mean()
```

## Proposed MLP Contract

The current `probing/mlp.py` file is empty, so the model contract still needs to
be implemented. A practical first contract is:

```text
input:  x, shape (B, 4096), float32
output: F_raw, shape (B, 9), float32
view:   F_pred = F_raw.reshape(B, 3, 3)
```

Recommended normalization/constraints to decide before training:

```text
F scale normalization:
  Sampson error is scale-invariant in theory, but numerical stability improves
  if F_pred is normalized, for example by Frobenius norm.

Rank-2 constraint:
  A fundamental matrix should have rank 2. This can be enforced by projecting
  F through SVD and zeroing the smallest singular value, or added later as a
  regularizer/diagnostic.

Coordinate scale:
  Stored correspondences are original ETH3D/COLMAP pixel coordinates. If training
  uses resized coordinates, transform points or F consistently before loss.
```

## End-to-End Diagram

```text
ETH3D scene/
  dslr_calibration_undistorted/images.txt
    -> ImageRecord objects with image pose, image name, 2D points, POINT3D_IDs
    -> pair correspondences from shared POINT3D_IDs
    -> targets.npz

ETH3D scene/images/...
    -> load_and_preprocess_images([image_id1, image_id2])
    -> VGGT model(images)
    -> aggregator forward hook captures camera tokens
    -> per-layer token arrays (1, 2, 2048)
    -> concat image-token 0 and image-token 1
    -> X_{split}_layer_{L}.npy rows (4096,)

manifest.csv
    -> maps each layer sample to:
       split, layer_idx, x_row, scene, image_id1, image_id2

Training MLP_L
    -> load X_train_layer_L.npy
    -> use manifest row to find targets.npz key
    -> MLP_L(x_row) predicts F_pred
    -> Sampson(F_pred, pts1, pts2)
    -> update only MLP_L
```

## Files That Matter For Training

```text
probing_experiment/scripts/build_probe_dataset.py
  Builds manifest.csv, X arrays, and targets.npz.

probing_experiment/src/probing_experiment/data_loading/dataset.py
  Loads one split/layer view of the artifacts.

probing_experiment/src/probing_experiment/probing/loss.py
  Implements Sampson error.

probing_experiment/src/probing_experiment/probing/mlp.py
  Currently empty; intended home for the layer-specific MLP module.

probing_experiment/src/probing_experiment/probing/train.py
  Currently empty or not yet wired; intended home for the training loop.
```
