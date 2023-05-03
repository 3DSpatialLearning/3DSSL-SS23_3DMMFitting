from utils.data import load_batch_data, load_camera_data
import os

main_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(str(main_dir))

LANDMARKS_DIR = "../data/toy_task/multi_frame_rgbd_fitting/landmarks"
CAMERA_DIR = "../data/toy_task/multi_frame_rgbd_fitting/222200037"

CONSISTENCY_DIR = CAMERA_DIR + "/colmap_consistency"
DEPTH_DIR = CAMERA_DIR + "/colmap_depth/"
NORMAL_DIR = CAMERA_DIR + "/colmap_normals/"
IMAGE_DIR = CAMERA_DIR + "/images/"
EXTRINSICS_FILE = CAMERA_DIR + "/extrinsics.npy"
INTRINSICS_FILE = CAMERA_DIR + "/intrinsics.npy"

SCALE = 1./1000.

if __name__ == '__main__':
    print("Loading frames data...")
    frames_data = load_batch_data(scale=SCALE,
                            landmarks_dir=LANDMARKS_DIR,
                            points_dir=DEPTH_DIR,
                            normals_dir=NORMAL_DIR,
                            image_dir=IMAGE_DIR)
    print("Loading camera data...")
    data = load_camera_data(extrinsics_path=EXTRINSICS_FILE, intrinsics_path=INTRINSICS_FILE)
    data['frames'] = frames_data

