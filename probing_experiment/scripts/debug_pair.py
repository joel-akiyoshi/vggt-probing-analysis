from pathlib import Path
import torch

from probing_experiment.data_loading.colmap_images import load_images_txt
from probing_experiment.data_loading.correspondences import get_all_image_pair_correspondences
from vggt.utils.load_fn import load_and_preprocess_images

from probing_experiment.vggt_utils.model import load_vggt
from probing_experiment.vggt_utils.inference import run_vggt_with_camera_token_cache
from probing_experiment.vggt_utils.debug import verify_vggt_outputs
from probing_experiment.vggt_utils.pointcloud import save_vggt_pointcloud_ply
from probing_experiment.artifact_io.activations import save_camera_token_activations_npz


def main():
    # Run VGGT and extract camera tokens for a single pair

    scene_name = "courtyard"
    scene_dir = Path("data/highres_train") / scene_name
    
    print("Loading images and correspondences...")
    image_records = load_images_txt(scene_dir / "dslr_calibration_undistorted" / "images.txt")
    pair_corrs = get_all_image_pair_correspondences(image_records, min_matches=100)

    scene_root = scene_dir / "images"
    image_1_id, image_2_id = next(iter(pair_corrs))

    im1_path = scene_root / image_records[image_1_id].name
    im2_path = scene_root / image_records[image_2_id].name

    print("Loading vggt...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_vggt(device)

    images = load_and_preprocess_images([im1_path, im2_path]).to(device)

    print("Running vggt...")
    predictions, activation_cache = run_vggt_with_camera_token_cache(model, images, device)

    verify_vggt_outputs(images, predictions, activation_cache)

    out_dir = Path("../outputs/vggt_debug_outputs")
    save_vggt_pointcloud_ply(
        predictions,
        out_dir / f"pair_{image_1_id}_{image_2_id}_vggt_world_points.ply",
        conf_quantile=0.1,
        max_points=300_000,
    )
    print(f"pointcloud written to {out_dir}")

    save_camera_token_activations_npz(
        activation_cache,
        out_dir / f"pair_{image_1_id}_{image_2_id}_camera_tokens.npz",
    )
    print(f".npz file written to {out_dir}")


if __name__ == "__main__":
    main()