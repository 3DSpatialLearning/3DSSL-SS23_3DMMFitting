import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.CameraFrameDataset import CameraFrameDataset
from utils.data import dict_tensor_to_np
from utils.transform import backproject_points
from models.LandmarkDetector import DlibLandmarkDetector
import os
import cv2
import pyvista as pv

main_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(str(main_dir))

DATA_DIR = "../data/toy_task/multi_frame_rgbd_fitting"
DLIB_DETECTOR_PATH = "../data/checkpoints/mmod_human_face_detector.dat"
DLIB_PREDICTOR_PATH = "../data/checkpoints/shape_predictor_68_face_landmarks.dat"

VISUALIZE_2D = False
VISUALIZE_LANDMARKS_3D = False

if __name__ == '__main__':
    print("Loading camera sequence data...")
    dataset = CameraFrameDataset(path_to_cam_dir=DATA_DIR, has_gt_landmarks=True)
    data_loader = DataLoader(
        dataset=dataset,
        pin_memory=True,
        batch_size=1,
    )
    print("Loading landmark detector...")
    landmark_detector = DlibLandmarkDetector(path_to_dlib_predictor_model=DLIB_PREDICTOR_PATH)
    print("Calculating landmarks...")
    for id, frame in enumerate(data_loader):
        frame = dict_tensor_to_np(frame)
        image = frame['image'].squeeeze()
        landmarks_2d = landmark_detector(image)
        if VISUALIZE_2D:
            cv2.namedWindow(f"image {id}", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(f"image {id}", int(image.shape[0] * 0.75), int(image.shape[1] * 0.75))
            for i, uv in enumerate(landmarks_2d):
                cv2.putText(image, str(i), tuple(uv), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(0, 0, 255), thickness=2)
            cv2.imshow(f"image {id}", image)
            cv2.waitKey(0)
            cv2.destroyWindow(f"image {id}")

        gt_landmarks = frame['gt_landmark'].squeeeze()
        estimated_landmarks = backproject_points(landmarks_2d, frame['depth'].squeeeze(), frame['intrinsics'].squeeeze(), frame['extrinsics'].squeeeze())

        if VISUALIZE_LANDMARKS_3D:
            not_nan_indices = ~(np.isnan(gt_landmarks).any(axis=1))
            criterion = nn.MSELoss(reduction='mean')
            error = criterion(torch.from_numpy(estimated_landmarks[not_nan_indices]),
                              torch.from_numpy(gt_landmarks[not_nan_indices]))
            plotter = pv.Plotter()
            plotter.add_mesh(pv.PolyData(gt_landmarks), color='red', point_size=5)
            plotter.add_mesh(pv.PolyData(estimated_landmarks), color='green', point_size=5)
            plotter.add_text(f"Landmarks (Red: Ground Truth, Green: Estimated). Mean square error {error.item()} ")
            plotter.add_axes(line_width=3, color='white', labels_off=False)
            plotter.camera_position = 'xy'
            plotter.show()

        exit(0)

