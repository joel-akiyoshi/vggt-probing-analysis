import torch


def verify_vggt_outputs(images: torch.Tensor, predictions: dict, activation_cache: dict):
    """
    Print sanity checks to verify the images passed through VGGT correctly.
    
    Takes in the input images tensor, output predictions dict, and the aggregator cache.
    Input images are expected to be resultant from load_and_preprocess_images().
    """
    
    # Input checks
    print("\n--- VGGT input sanity checks ---")
    print(f"images shape: {tuple(images.shape)}  dtype={images.dtype}  device={images.device}")

    # Output checks
    print("\n--- prediction keys/shapes ---")
    for k, v in predictions.items():
        if torch.is_tensor(v):
            print(f"{k:20s} shape={tuple(v.shape)} dtype={v.dtype}")
        elif isinstance(v, list):
            print(f"{k:20s} list length={len(v)}")
        else:
            print(f"{k:20s} type={type(v)}")

    # Check all fields
    required = ["pose_enc", "depth", "depth_conf", "world_points", "world_points_conf"]
    missing = [k for k in required if k not in predictions]
    if missing:
        raise RuntimeError(f"VGGT did not return expected keys: {missing}")

    # Check camera token shapes, should be (1, 2, 2048)
    print("\n--- camera-token activation sanity checks ---")
    cam_tokens = activation_cache.get("camera_tokens", {})
    print(f"cached camera-token layers: {sorted(cam_tokens.keys())}")
    
    for layer_idx, acts in cam_tokens.items():
        print(f"layer {layer_idx:02d}: camera token shape {tuple(acts.shape)}")
