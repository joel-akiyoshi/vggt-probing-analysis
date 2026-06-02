from src.data_loading.colmap_images import load_images_txt
from src.data_loading.correspondences import get_all_image_pair_correspondences

from pathlib import Path
import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images



# courtyard scene
image_records = load_images_txt("highres_train/courtyard/dslr_calibration_undistorted/images.txt")
pair_corrs = get_all_image_pair_correspondences(image_records, min_matches=100)
# print_correspondence_summary(image_records, pair_corrs)

# one pair test
scene_root = Path("highres_train/courtyard/images")
image_1_id, image_2_id = next(iter(pair_corrs))
im1_path = scene_root / image_records[image_1_id].name
im2_path = scene_root / image_records[image_2_id].name
print(f"Using image pair:\n  {im1_path}\n  {im2_path}")

# feed a pair through VGGT
device = "cuda" if torch.cuda.is_available() else "cpu"


images = load_and_preprocess_images([im1_path, im2_path]).to(device)

# Run inference and capture camera-token activations from the aggregator.
print("Running inference...")
if device == "cuda":
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
else:
    dtype = torch.float32

hook_handle, activation_cache = register_aggregator_camera_token_hook(model)
try:
    with torch.no_grad():
        if device == "cuda":
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
        else:
            predictions = model(images)
finally:
    hook_handle.remove()

verify_vggt_outputs(images, predictions, activation_cache)

out_dir = Path("vggt_debug_outputs")
save_vggt_pointcloud_ply(
    predictions,
    out_dir / f"pair_{image_1_id}_{image_2_id}_vggt_world_points.ply",
    conf_quantile=0.1,
    max_points=300_000_000,
)
save_camera_token_activations_npz(
    activation_cache,
    out_dir / f"pair_{image_1_id}_{image_2_id}_camera_tokens.npz",
)

X = np.load("vggt_debug_outputs/pair_1_2_camera_tokens.npz")
print(X.keys())
print(X.items())


# this hooks out the camera tokens. 
# 1) Understand this a bit better. DONE
# 2) Start working on scaling up, decide on dimensionality for training, testing. 
# 3) epipolar sampson stuff
# 4) folder organization

# Scaling up
    # PER LAYER, a single MLP:
    # choose how to organize X_tr and y_tr, X_te and y_te
    # X_tr: shape (n, 4096)
        # each of the n rows represents an image pair. The 4096 dim feature vector is constructed from the activations
        # use smart combinations (concatenations, element wise products, asymmetric concatenation, etc), or just start with plain concat
    # y_tr: shape (n, ImageRecord: specifically use the correspondences)
        # each of the n rows contains the feature correspondences to use with the sampson error
    # compute_loss(pts1, pts2, pred_K)

