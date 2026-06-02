from pathlib import Path
import torch
import numpy as np


# consolidated logic from vggt's demo_gradio.py
def save_vggt_pointcloud_ply(
    predictions: dict,
    out_path: str | Path,
    conf_quantile: float = 0.50,
    max_points: int = 300_000,
):
    """Save VGGT predicted world_points as an ASCII PLY point cloud.

    Uses predictions['world_points'] as XYZ and predictions['images'] as RGB.
    Filters by world_points_conf >= quantile(conf_quantile), then uniformly downsamples.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    points = predictions["world_points"].detach().float().cpu()        # [B,S,H,W,3]
    conf = predictions["world_points_conf"].detach().float().cpu()     # [B,S,H,W]
    imgs = predictions["images"].detach().float().cpu()                # [B,S,3,H,W]

    if points.ndim != 5 or points.shape[-1] != 3:
        raise ValueError(f"Expected world_points [B,S,H,W,3], got {tuple(points.shape)}")
    if imgs.ndim != 5:
        raise ValueError(f"Expected images [B,S,3,H,W], got {tuple(imgs.shape)}")

    # [B,S,3,H,W] -> [B,S,H,W,3]
    colors = imgs.permute(0, 1, 3, 4, 2).clamp(0, 1)

    valid = torch.isfinite(points).all(dim=-1) & torch.isfinite(conf)
    threshold = torch.quantile(conf[valid], conf_quantile) if valid.any() else torch.tensor(float("inf"))
    keep = valid & (conf >= threshold)

    points = points[keep].numpy()
    colors = (colors[keep].numpy() * 255.0).astype(np.uint8)

    if points.shape[0] == 0:
        raise RuntimeError("No valid points survived filtering; lower conf_quantile.")

    if points.shape[0] > max_points:
        idx = np.linspace(0, points.shape[0] - 1, max_points).astype(np.int64)
        points = points[idx]
        colors = colors[idx]

    with out_path.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")

    print(f"Saved {points.shape[0]} points to {out_path}")
    return out_path
