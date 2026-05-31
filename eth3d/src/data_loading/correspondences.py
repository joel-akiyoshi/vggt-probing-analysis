from __future__ import annotations

from itertools import combinations
from typing import Dict, Tuple
from .colmap_images import ImageRecord

import numpy as np


# Type aliases for readability
ImageId = int
Point3DId = int

# A single observation is:
#     (image_id, x, y)
Observation = Tuple[ImageId, float, float]

# A pairwise correspondence output is:
#     pts1: np.ndarray of shape (N, 2)
#     pts2: np.ndarray of shape (N, 2)
#     point3d_ids: np.ndarray of shape (N,)
CorrespondenceResult = Tuple[np.ndarray, np.ndarray, np.ndarray]


def get_valid_observations(image_record: ImageRecord) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract valid 2D points and their POINT3D_IDs from an ImageRecord.
    Valid if [x, y, points3d_id] does not have -1 as points3d_id.

    Args:
        image_record:
            An ImageRecord object with a points2d field of shape (N, 3).
            Each row is [x, y, point3d_id].

    Returns:
        xy:
            np.ndarray of shape (M, 2), containing valid 2D points.

        point3d_ids:
            np.ndarray of shape (M,), containing corresponding POINT3D_IDs.
    """
    points2d = image_record.points2d

    if points2d.ndim != 2 or points2d.shape[1] != 3:
        raise ValueError(
            f"Expected points2d to have shape (N, 3), got {points2d.shape}"
        )

    # filter out 2d keypoints that do not map to any points3d id
    valid_mask = points2d[:, 2] != -1

    valid_points = points2d[valid_mask]
    xy = valid_points[:, :2].astype(np.float64)
    point3d_ids = valid_points[:, 2].astype(np.int64)

    return xy, point3d_ids


def build_image_point3d_map(image_record: ImageRecord) -> Dict[Point3DId, np.ndarray]:
    """
    Build a dictionary mapping POINT3D_ID -> 2D pixel location for one image.

    Args:
        image_record:
            An ImageRecord object.

    Returns:
        Dictionary mapping point3d_id -> np.ndarray([x, y])

    Example:
        {
            28390: np.array([2078.83, 680.647]),
            30028: np.array([2210.27, 789.918]),
        }
    """
    xy, point3d_ids = get_valid_observations(image_record)

    point_map: Dict[Point3DId, np.ndarray] = {}

    for point_xy, point3d_id in zip(xy, point3d_ids):
        point3d_id = int(point3d_id)

        if point3d_id not in point_map:
            point_map[point3d_id] = point_xy

    return point_map


def get_correspondences(
    images: Dict[ImageId, ImageRecord],
    image_id1: ImageId,
    image_id2: ImageId
) -> CorrespondenceResult:
    """
    Get 2D-2D correspondences between two images using shared POINT3D_IDs.

    Args:
        images: Dictionary mapping image_id -> ImageRecord.

        image_id1: First image ID.

        image_id2: Second image ID.

    Returns:
        pts1:
            np.ndarray of shape (N, 2), 2D points from image 1.

        pts2:
            np.ndarray of shape (N, 2), matching 2D points from image 2.

        shared_point3d_ids:
            np.ndarray of shape (N,), shared POINT3D_IDs.

    Example:
        pts1, pts2, ids = get_correspondences(images, 38, 37)

        pts1[i] and pts2[i] are observations of the same 3D point ids[i].
    """
    if image_id1 not in images:
        raise KeyError(f"image_id1={image_id1} not found in images dictionary.")

    if image_id2 not in images:
        raise KeyError(f"image_id2={image_id2} not found in images dictionary.")

    # extract ImageRecord at that index
    image1 = images[image_id1]
    image2 = images[image_id2]
 
    # get each image's mappings from {3d world point id: [x, y] point in the image} for all mapped points
    map1 = build_image_point3d_map(image1)
    map2 = build_image_point3d_map(image2)

    # shared ids are the 3d world point ids that appear in both maps
    shared_ids = sorted(set(map1.keys()) & set(map2.keys()))

    # pts is (N, 2) storing the [x, y] coordinates for the shared points
    pts1 = np.array([map1[pid] for pid in shared_ids], dtype=np.float64)
    pts2 = np.array([map2[pid] for pid in shared_ids], dtype=np.float64)
    shared_point3d_ids = np.array(shared_ids, dtype=np.int64)

    return pts1, pts2, shared_point3d_ids


def get_all_image_pair_correspondences(
    images: Dict[ImageId, object],
    min_matches: int = 0,
) -> Dict[Tuple[ImageId, ImageId], CorrespondenceResult]:
    """
    Compute correspondences for every image pair.

    Args:
        images:
            Dictionary mapping image_id -> ImageRecord.

        min_matches:
            Only keep image pairs with at least this many correspondences.

    Returns:
        Dictionary mapping: (image_id1, image_id2) -> (pts1, pts2, point3d_ids)
            - pts1.shape == (N, 2)
            - pts2.shape == (N, 2)
            - point3d_ids.shape == (N,)
    """
    pair_correspondences: Dict[Tuple[ImageId, ImageId], CorrespondenceResult] = {}

    image_ids = sorted(images.keys())  # lexicographical sort

    for image_id1, image_id2 in combinations(image_ids, 2):
        pts1, pts2, point3d_ids = get_correspondences(
            images,
            image_id1,
            image_id2
        )

        if len(point3d_ids) >= min_matches:
            pair_correspondences[(image_id1, image_id2)] = (
                pts1,
                pts2,
                point3d_ids,
            )

    return pair_correspondences


def print_correspondence_summary(
    images: Dict[ImageId, object],
    pair_correspondences: Dict[Tuple[ImageId, ImageId], CorrespondenceResult],
    max_pairs: int = 10,
) -> None:
    """
    Print a small summary of pairwise correspondences.

    Args:
        images:
            Dictionary mapping image_id -> ImageRecord.

        pair_correspondences:
            Output from get_all_image_pair_correspondences() or
            build_pair_correspondences_from_index().

        max_pairs:
            Maximum number of image pairs to print.
    """
    print(f"Found correspondences for {len(pair_correspondences)} image pairs.")

    sorted_pairs = sorted(
        pair_correspondences.items(),
        key=lambda item: item[1][0].shape[0],
        reverse=True,
    )

    for i, ((image_id1, image_id2), (pts1, pts2, point3d_ids)) in enumerate(
        sorted_pairs[:max_pairs]
    ):
        name1 = getattr(images[image_id1], "name", str(image_id1))
        name2 = getattr(images[image_id2], "name", str(image_id2))

        print()
        print(f"Pair {i + 1}")
        print(f"  Image {image_id1}: {name1}")
        print(f"  Image {image_id2}: {name2}")
        print(f"  Matches: {len(point3d_ids)}")
        print(f"  pts1 shape: {pts1.shape}")
        print(f"  pts2 shape: {pts2.shape}")
