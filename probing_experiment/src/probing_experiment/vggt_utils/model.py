from vggt.models.vggt import VGGT
import torch


def load_vggt(device: str):
    """Load vggt model from pretrained weights."""
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model = VGGT()
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)
    return model
