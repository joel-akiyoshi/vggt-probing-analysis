from pathlib import Path
import numpy as np


def save_camera_token_activations_npz(activation_cache: dict, out_path: str | Path):
    """
    Save camera token activations for MLP probing. Converting tensors to numpy.

    Args:
        activation_cache: holds the camera tokens extracted by the hook (tensors)

        out_path: the file location to write the file
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    arrays = {}

    # extract camera activation tensors and convert to numpy
    for layer_idx, acts in activation_cache["camera_tokens"].items():
        arrays[f"camera_layer_{layer_idx}"] = acts.numpy()  # shape (1, 2, 2048)
    
    # extract the layer numbers that were cached
    arrays["cached_layers"] = np.array(sorted(activation_cache["camera_tokens"].keys()), dtype=np.int64)

    # save the camera activations and layer numbers
    np.savez(out_path, **arrays)
    print(f"Saved activation arrays to {out_path}")


def load_camera_token_activations_npz(npz_path: str | Path):
    """
    Load the camera token activations into dict.

    Returns:
        activation_tokens: dict mapping {"cached_layers" -> numpy array of layer numbers,
                                          "camera_layer_1" -> camera token activation arrays for L1,
                                          "camera_layer_2" -> ...}
        
    """ 
    return np.load(npz_path)
