from src.data_loading.colmap_images import load_images_txt
from src.data_loading.correspondences import get_correspondences, get_all_image_pair_correspondences, print_correspondence_summary


images = load_images_txt("highres_train/courtyard/dslr_calibration_undistorted/images.txt")

# Single image pair
pts1, pts2, point3d_ids = get_correspondences(images, 38, 37)

print(pts1.shape)
print(pts2.shape)
print(point3d_ids.shape)

# All image pairs
pair_corrs = get_all_image_pair_correspondences(images, min_matches=1000)

print_correspondence_summary(images, pair_corrs)