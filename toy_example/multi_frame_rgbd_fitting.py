from utils.data import load_batch_data, load_camera_data, resize_data_images
from models.LandmarkDetector import DlibLandmarkDetector
import os
import cv2

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

DLIB_DETECTOR_PATH = "../data/checkpoints/mmod_human_face_detector.dat"
DLIB_PREDICTOR_PATH = "../data/checkpoints/shape_predictor_68_face_landmarks.dat"

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
    print("Loading landmark detector...")
    detector = DlibLandmarkDetector(DLIB_DETECTOR_PATH, DLIB_PREDICTOR_PATH)

    print("Visualizing frames...")
    for id, frame in data['frames'].items():
        frame = data['frames'][id]
        image = frame['image']
        landmarks = detector.detect_landmarks(image)
        cv2.namedWindow(f"image {id}", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(f"image {id}", int(image.shape[0]*0.25), int(image.shape[1]*0.25))
        for point in landmarks:
            cv2.circle(image, tuple(point), radius=10, color=(0, 0, 255), thickness=-1)
        cv2.imshow(f"image {id}", image)
        cv2.waitKey(0)
        cv2.destroyWindow(f"image {id}")


