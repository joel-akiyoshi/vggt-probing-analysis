from contextlib import nullcontext
import torch

from probing_experiment.vggt_utils.hooks import register_aggregator_camera_token_hook


# logic from demo_gradio.py with inserted hook
def run_vggt_with_camera_token_cache(model, images, device):
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)
    else:
        autocast_ctx = nullcontext()

    handle, activation_cache = register_aggregator_camera_token_hook(model)

    try:
        with torch.no_grad():
            with autocast_ctx:
                predictions = model(images)
    finally:
        handle.remove()  # always remove hook

    return predictions, activation_cache
