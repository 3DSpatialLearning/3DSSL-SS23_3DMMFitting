from utils.data import load_batch_data, load_camera_data
from utils.transform import backproject_points
from models.LandmarkDetector import DlibLandmarkDetector
import os
import cv2
import pyvista as pv

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

VISUALIZE = False
VISUALIZE_LANDMARKS_3D = True

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
    data["extrinsics"][:, 3] *= 0.001
    print("Loading landmark detector...")
    detector = DlibLandmarkDetector(None, DLIB_PREDICTOR_PATH)

    print("Calculating landmarks...")
    for id, frame in data['frames'].items():
        frame = data['frames'][id]
        image = frame['image']
        landmarks_2d = detector.detect_landmarks(image)
        if VISUALIZE:
            cv2.namedWindow(f"image {id}", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(f"image {id}", int(image.shape[0]*0.75), int(image.shape[1]*0.75))
            for i, uv in enumerate(landmarks_2d):
                cv2.putText(image, str(i), tuple(uv), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(0, 0, 255), thickness=2)
            cv2.imshow(f"image {id}", image)
            cv2.waitKey(0)
            cv2.destroyWindow(f"image {id}")
        print(data['extrinsics'], data['intrinsics'])
        gt_landmarks = frame['landmarks']
        estimated_landmarks = backproject_points(landmarks_2d, frame['points'], data['intrinsics'], data['extrinsics'])
        
        if VISUALIZE_LANDMARKS_3D:
            plotter = pv.Plotter()
            plotter.add_mesh(pv.PolyData(gt_landmarks), color='red', point_size=7)
            plotter.add_mesh(pv.PolyData(estimated_landmarks), color='green', point_size=7)
            plotter.add_text("Landmarks (Red: Ground Truth, Green: Estimated)")
            plotter.add_axes(line_width=5, labels_off=False)
            plotter.camera_position = 'xy'
            plotter.show()

        print(estimated_landmarks[:15])
        print('--------------------------------------------------')
        print(gt_landmarks[:15])
        exit(0)

