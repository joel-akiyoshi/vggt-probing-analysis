from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np


@dataclass
class ImageRecord:
    image_id: int
    qvec: np.ndarray        # shape (4,) -> [qw, qx, qy, qz]
    tvec: np.ndarray        # shape (3,) -> [tx, ty, tz]
    camera_id: int
    name: str
    points2d: np.ndarray    # shape (N, 3) -> rows are [x, y, point3d_id]


def parse_metadata_line(line: str) -> tuple:
    """
    Parse a COLMAP metadata line, within the images.txt file.
    
    Formatted as: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME

    Returns:
        image_id, qvec (numpy 4-vector), tvec (numpy 3-vector), camera_id, name

    """
    
    metadata = line.split()

    if len(metadata) < 10:
        raise ValueError(f"Invalid image metadata line:\n{line}")
    
    image_id = int(metadata[0])
    qvec = np.array([float(metadata[1]), float(metadata[2]), float(metadata[3]), float(metadata[4])], dtype=np.float64)
    tvec = np.array([float(metadata[5]), float(metadata[6]), float(metadata[7])], dtype=np.float64)
    camera_id = int(metadata[8])
    name = "_".join(metadata[9:])  # in case of images names with spaces, join them together with an underscore

    return image_id, qvec, tvec, camera_id, name
    
    
def parse_points2d_line(line: str) -> np.ndarray:
    """
    Parse a COLMAP POINTS2D line, within the images.txt file.
    
    Formatted as: X Y POINT3D_ID X Y POINT3D_ID

    Returns:
        np.ndarray of shape (N, 3), where each row is:
        [x, y, point3d_id]
    """
    tokens = line.split()

    if len(tokens) % 3 != 0:
        raise ValueError(
            f"POINTS2D line has {len(tokens)} values, which is not divisible by 3."
        )

    points = []
    for i in range(0, len(tokens), 3):
        x = float(tokens[i])
        y = float(tokens[i + 1])
        point3d_id = int(tokens[i + 2])
        points.append([x, y, point3d_id])

    return np.array(points, dtype=np.float64)


def load_images_txt(path: str | Path) -> Dict[int, ImageRecord]:
    """
    Load a COLMAP images.txt file into a dictionary.

    Args:
        path: Path to images.txt.

    Returns:
        Dictionary of {image_id : ImageRecord}
    """

    path = Path(path)

    with path.open("r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Remove comment/header lines
    data_lines = [line for line in lines if not line.startswith("#")]

    if len(data_lines) % 2 != 0:
        raise ValueError(
            f"Expected an even number of data lines, but got {len(data_lines)}."
        )

    images: Dict[int, ImageRecord] = {}

    # iterate over even numbered lines
    for i in range(0, len(data_lines), 2):
        metadata_line = data_lines[i]
        points2d_line = data_lines[i + 1]

        image_id, qvec, tvec, camera_id, name = parse_metadata_line(metadata_line)
        points2d = parse_points2d_line(points2d_line)

        images[image_id] = ImageRecord(
            image_id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=name,
            points2d=points2d,
        )

    return images


def summarize_images(images: Dict[int, ImageRecord]) -> None:
    """
    Print a quick summary to verify loading worked.
    """
    print(f"Loaded {len(images)} images")

    for image_id, record in list(images.items())[:5]:
        num_points = record.points2d.shape[0]
        num_valid = np.sum(record.points2d[:, 2] != -1)

        print()
        print(f"Image ID: {image_id}")
        print(f"Name: {record.name}")
        print(f"Camera ID: {record.camera_id}")
        print(f"qvec: {record.qvec}")
        print(f"tvec: {record.tvec}")
        print(f"2D observations: {num_points}")
        print(f"Valid 3D-linked observations: {num_valid}")


if __name__ == "__main__":
    images_path = "../../highres_train/courtyard/dslr_calibration_undistorted/images.txt"

    images = load_images_txt(images_path)
    summarize_images(images)
