import torch
from pathlib import Path

class ETH3DPairDataset(torch.utils.data.Dataset):
    def __init__(self, scene_root, images, pair_corrs, transform=None):
        self.scene_root = Path(scene_root)
        self.images = images
        self.pair_items = list(pair_corrs.items())
        self.transform = transform

    def __len__(self):
        return len(self.pair_items)

    def __getitem__(self, idx):
        (image_id1, image_id2), (pts1, pts2, point3d_ids) = self.pair_items[idx]

        img1_path = self.scene_root / self.images[image_id1].name
        img2_path = self.scene_root / self.images[image_id2].name

        img1 = load_image(img1_path)
        img2 = load_image(img2_path)

        img_pair = preprocess_for_vggt([img1, img2])

        return {
            "image_ids": (image_id1, image_id2),
            "image_paths": (str(img1_path), str(img2_path)),
            "images": img_pair,
            "pts1": pts1,
            "pts2": pts2,
            "point3d_ids": point3d_ids,
        }