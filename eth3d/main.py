from src.data_loading.colmap_images import load_images_txt
from src.data_loading.correspondences import get_all_image_pair_correspondences, print_correspondence_summary

from pathlib import Path
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import torch


# courtyard scene
images = load_images_txt("highres_train/courtyard/dslr_calibration_undistorted/images.txt")
pair_corrs = get_all_image_pair_correspondences(images, min_matches=100)
# print_correspondence_summary(images, pair_corrs)

# one pair test
scene_root = Path("highres_train/courtyard/images")
image_1_id, image_2_id = next(iter(pair_corrs))
im1_path = scene_root / images[image_1_id].name
im2_path = scene_root / images[image_2_id].name

# feed a pair through VGGT
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Initializing and loading VGGT model...")
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model = VGGT()
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
model.eval()
model = model.to(device)

images = load_and_preprocess_images([im1_path, im2_path]).to(device)

# Run inference
print("Running inference...")
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        predictions = model(images)

print(predictions.keys())


# load all scenes with their ImageRecords and correspondence pairs

# aggregate into training (11 scenes) and testing (2 scenes), load each into respective Dataset
    # Training dataset images, feed through VGGT, save the activations into an X_train array.
    # Testing dataset images, feed through VGGT, save the activations into a X_test array.

    # With X_train + GT correspondences: feed pred_f = MLP(X_train), backprop with loss(pred_f, GT_corr), iterate for 50 epochs
    # With X_test + GT correspondences: feed pred_f = MLP(X_test), evaluate with loss(pred_f, GT_corr)


# (separate module) sampson error calculation, accounting for image resizing

# (separate module) MLP

# (this module) pytorch forward() hook extraction