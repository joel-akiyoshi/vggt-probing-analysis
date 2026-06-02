from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch


def to_homogeneous(points: torch.Tensor) -> torch.Tensor:
    """Append a homogeneous 1 coordinate to (..., N, 2) point tensors."""
    if points.shape[-1] != 2:
        raise ValueError(f"Expected points with final dim 2, got {points.shape}")

    ones = torch.ones_like(points[..., :1])
    return torch.cat([points, ones], dim=-1)


def sampson_error(
    fundamental: torch.Tensor,
    pts1: torch.Tensor,
    pts2: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    First-order geometric reprojection error for corresponding image points.

    Args:
        fundamental:
            Fundamental matrix/matrices with shape (..., 3, 3), mapping points
            in image 1 to epipolar lines in image 2.
        pts1:
            Pixel coordinates from image 1 with shape (..., N, 2).
        pts2:
            Pixel coordinates from image 2 with shape (..., N, 2).
        eps:
            Small value preventing division by zero for degenerate inputs.

    Returns:
        Per-correspondence Sampson errors with shape broadcast from (..., N).
    """
    if fundamental.shape[-2:] != (3, 3):
        raise ValueError(f"Expected F with shape (..., 3, 3), got {fundamental.shape}")

    pts1_h = to_homogeneous(pts1)
    pts2_h = to_homogeneous(pts2)

    f_x1 = torch.matmul(fundamental, pts1_h.unsqueeze(-1)).squeeze(-1)
    ft_x2 = torch.matmul(
        fundamental.transpose(-1, -2), pts2_h.unsqueeze(-1)
    ).squeeze(-1)
    x2t_f_x1 = (pts2_h * f_x1).sum(dim=-1)

    denom = (
        f_x1[..., 0].square()
        + f_x1[..., 1].square()
        + ft_x2[..., 0].square()
        + ft_x2[..., 1].square()
    ).clamp_min(eps)

    return x2t_f_x1.square() / denom


def mean_sampson_error(
    fundamental: torch.Tensor,
    pts1: torch.Tensor,
    pts2: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Mean Sampson error over all correspondences and batch dimensions."""
    return sampson_error(fundamental, pts1, pts2, eps=eps).mean()


def scale_fundamental_matrix(
    fundamental: torch.Tensor,
    scale1: tuple[float, float],
    scale2: tuple[float, float],
) -> torch.Tensor:
    """
    Transform F when image coordinates are scaled.

    If resized coordinates are x1' = T1 x1 and x2' = T2 x2, the resized-image
    fundamental matrix is F' = T2^{-T} F T1^{-1}.
    """
    dtype = fundamental.dtype
    device = fundamental.device

    sx1, sy1 = scale1
    sx2, sy2 = scale2
    t1_inv = torch.tensor(
        [[1.0 / sx1, 0.0, 0.0], [0.0, 1.0 / sy1, 0.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )
    t2_inv = torch.tensor(
        [[1.0 / sx2, 0.0, 0.0], [0.0, 1.0 / sy2, 0.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )

    return t2_inv.transpose(-1, -2) @ fundamental @ t1_inv


@dataclass
class CameraRecord:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    @property
    def K(self) -> np.ndarray:
        if self.model == "PINHOLE":
            fx, fy, cx, cy = self.params[:4]
        elif self.model == "SIMPLE_PINHOLE":
            f, cx, cy = self.params[:3]
            fx = fy = f
        else:
            raise ValueError(f"Unsupported camera model for this test: {self.model}")

        return np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


def _load_cameras_txt(path: str | Path) -> Dict[int, CameraRecord]:
    cameras: Dict[int, CameraRecord] = {}
    with Path(path).open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            camera_id = int(tokens[0])
            cameras[camera_id] = CameraRecord(
                camera_id=camera_id,
                model=tokens[1],
                width=int(tokens[2]),
                height=int(tokens[3]),
                params=np.array([float(x) for x in tokens[4:]], dtype=np.float64),
            )
    return cameras


def _load_points3d_xyz(path: str | Path, point_ids: Iterable[int]) -> Dict[int, np.ndarray]:
    wanted = {int(point_id) for point_id in point_ids}
    points: Dict[int, np.ndarray] = {}
    with Path(path).open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            point_id = int(tokens[0])
            if point_id in wanted:
                points[point_id] = np.array(
                    [float(tokens[1]), float(tokens[2]), float(tokens[3])],
                    dtype=np.float64,
                )
                if len(points) == len(wanted):
                    break
    return points


def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec / np.linalg.norm(qvec)
    return np.array(
        [
            [
                1.0 - 2.0 * qy * qy - 2.0 * qz * qz,
                2.0 * qx * qy - 2.0 * qz * qw,
                2.0 * qx * qz + 2.0 * qy * qw,
            ],
            [
                2.0 * qx * qy + 2.0 * qz * qw,
                1.0 - 2.0 * qx * qx - 2.0 * qz * qz,
                2.0 * qy * qz - 2.0 * qx * qw,
            ],
            [
                2.0 * qx * qz - 2.0 * qy * qw,
                2.0 * qy * qz + 2.0 * qx * qw,
                1.0 - 2.0 * qx * qx - 2.0 * qy * qy,
            ],
        ],
        dtype=np.float64,
    )


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _fundamental_from_colmap(image1, image2, camera1: CameraRecord, camera2: CameraRecord):
    r1 = _qvec_to_rotmat(image1.qvec)
    r2 = _qvec_to_rotmat(image2.qvec)
    t1 = image1.tvec.reshape(3)
    t2 = image2.tvec.reshape(3)

    r21 = r2 @ r1.T
    t21 = t2 - r21 @ t1
    essential = _skew(t21) @ r21
    fundamental = np.linalg.inv(camera2.K).T @ essential @ np.linalg.inv(camera1.K)

    norm = np.linalg.norm(fundamental)
    if norm > 0:
        fundamental = fundamental / norm
    return fundamental


def _project_points(points3d: np.ndarray, image, camera: CameraRecord) -> np.ndarray:
    r = _qvec_to_rotmat(image.qvec)
    points_cam = (r @ points3d.T).T + image.tvec.reshape(1, 3)
    pixels_h = (camera.K @ points_cam.T).T
    return pixels_h[:, :2] / pixels_h[:, 2:3]


def _run_basic_tests() -> None:
    f = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    pts1 = torch.tensor([[2.0, 3.0], [4.0, -1.0], [0.5, 0.25]], dtype=torch.float64)
    pts2 = torch.tensor([[-5.0, 3.0], [8.0, -1.0], [10.0, 0.25]], dtype=torch.float64)
    err = sampson_error(f, pts1, pts2)
    assert torch.allclose(err, torch.zeros_like(err), atol=1e-12), err
    print(f"Synthetic zero-error test max: {err.max().item():.3e}")


def _run_eth3d_test() -> None:
    from probing_experiment.data_loading.colmap_images import load_images_txt
    from probing_experiment.data_loading.correspondences import get_correspondences

    experiment_root = Path(__file__).resolve().parents[3]
    calibration_root = (
        experiment_root
        / "data/highres_train/courtyard/dslr_calibration_undistorted"
    )

    images = load_images_txt(calibration_root / "images.txt")
    image_id1, image_id2 = 28, 38
    observed_pts1, observed_pts2, point3d_ids = get_correspondences(
        images, image_id1, image_id2
    )

    cameras = _load_cameras_txt(calibration_root / "cameras.txt")
    image1 = images[image_id1]
    image2 = images[image_id2]
    camera1 = cameras[image1.camera_id]
    camera2 = cameras[image2.camera_id]
    fundamental = _fundamental_from_colmap(image1, image2, camera1, camera2)

    point3d_ids = point3d_ids[:256]
    points3d_by_id = _load_points3d_xyz(calibration_root / "points3D.txt", point3d_ids)
    ordered_ids = [point_id for point_id in point3d_ids if int(point_id) in points3d_by_id]
    points3d = np.stack([points3d_by_id[int(point_id)] for point_id in ordered_ids])

    projected_pts1 = _project_points(points3d, image1, camera1)
    projected_pts2 = _project_points(points3d, image2, camera2)

    f_t = torch.from_numpy(fundamental)
    proj_err = sampson_error(
        f_t, torch.from_numpy(projected_pts1), torch.from_numpy(projected_pts2)
    )
    assert proj_err.max().item() < 1e-18, proj_err.max().item()
    print(
        "ETH3D projected ground-truth zero-error test max: "
        f"{proj_err.max().item():.3e}"
    )

    obs_err = sampson_error(
        f_t, torch.from_numpy(observed_pts1[: len(ordered_ids)]), torch.from_numpy(observed_pts2[: len(ordered_ids)])
    )
    print(
        "ETH3D observed COLMAP keypoint Sampson error "
        f"mean={obs_err.mean().item():.3e}, max={obs_err.max().item():.3e}"
    )


if __name__ == "__main__":
    _run_basic_tests()
    _run_eth3d_test()
