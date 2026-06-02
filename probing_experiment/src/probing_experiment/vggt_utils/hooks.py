from vggt.models.vggt import VGGT


def register_aggregator_camera_token_hook(model: VGGT):
    """
    Capture VGGT aggregator camera-token activations during model(images).

    Aggregator outputs a list where cached layers contain tensors of shape
    [B, S, P, 2C]. Token index 0 is the camera token. 

    Returns:
        handle: hook handle, returned so it can be removed after hook executes.

        cache: stores the camera tokens (tensors), will be mutated as hook executes. 
    """
    cache = {}
    
    # https://docs.pytorch.org/docs/2.12/generated/torch.nn.modules.module.register_module_forward_hook.html
    def hook(_module, _inputs, output):

        # unpack the aggregator output, patch_start_idx unused
        aggregated_tokens_list, _ = output
        camera_tokens = {}

        for layer_idx, tokens in enumerate(aggregated_tokens_list):
            if tokens is None:
                print(f"layer index: {layer_idx} has no cached tokens")
                continue

            # tokens: [B, S, P, 2C]
            # aggregator.py specifies the P dimension as: [camera_tokens, register_tokens, patch_summary_tokens]. Camera token is elem 0.
            camera_tokens[layer_idx] = tokens[:, :, 0, :].detach().float().cpu()

        cache["camera_tokens"] = camera_tokens  # camera tokens is a dict with {layer_idx (int) : camera_token_activations (shape 1, 2, 2048)}

    handle = model.aggregator.register_forward_hook(hook)
    return handle, cache
