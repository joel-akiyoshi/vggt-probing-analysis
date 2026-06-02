import py7zr

def unzip_7z(archive_path, output_dir):
    """
    Extracts all files from a .7z archive.
    """
    try:
        with py7zr.SevenZipFile(archive_path, mode='r') as archive:
            archive.extractall(path=output_dir)
        print(f"Successfully extracted {archive_path} to {output_dir}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Usage
unzip_7z('multi_view_training_dslr_undistorted.7z', '../data/highres_train')
