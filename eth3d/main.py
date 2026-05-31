from src.data_loading.colmap_images import load_images_txt
from src.data_loading.correspondences import get_all_image_pair_correspondences, print_correspondence_summary


# courtyard scene
images = load_images_txt("highres_train/courtyard/dslr_calibration_undistorted/images.txt")
pair_corrs = get_all_image_pair_correspondences(images, min_matches=100)
print_correspondence_summary(images, pair_corrs)

# load all scenes with their ImageRecords and correspondence pairs
# aggregate into training and test (2 scenes)

# use the images and pair_corrs to access the filepaths

