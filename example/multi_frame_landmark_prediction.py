import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import cv2
import pyvista as pv

from dataset.CameraFrameDataset import CameraFrameDataset
from dataset.utils import dict_tensor_to_np
from models_.LandmarkDetector import DlibLandmarkDetector
from config import get_config

main_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(str(main_dir))

VISUALIZE_2D = True
VISUALIZE_LANDMARKS_3D = True

if __name__ == '__main__':
    print("Loading camera sequence data...")
    config = get_config(path_to_data_dir="../")
    dataset = CameraFrameDataset(path_to_data=config.cam_data_dir, has_gt_landmarks=True)
    print("Loading landmark detector...")
    landmark_detector = DlibLandmarkDetector(path_to_dlib_predictor_model=config.dlib_face_predictor_path)
    dataset.precompute_landmarks(landmark_detector, force_precompute=True)
    data_loader = DataLoader(
        dataset=dataset,
        pin_memory=True,
        batch_size=1,
    )
    for id, frame in enumerate(data_loader):
        frame = dict_tensor_to_np(frame)
        image = frame['image'].squeeze()
        if VISUALIZE_2D:
            landmarks_2d = frame['predicted_landmark_2d'].squeeze()
            cv2.namedWindow(f"image {id}", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(f"image {id}", int(image.shape[0] * 0.75), int(image.shape[1] * 0.75))
            for i, uv in enumerate(landmarks_2d):
                cv2.putText(image, str(i), tuple(uv), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(0, 0, 255), thickness=2)
            cv2.imshow(f"image {id}", image)
            cv2.waitKey(0)
            # cv2.destroyWindow(f"image {id}")

        if VISUALIZE_LANDMARKS_3D:
            gt_landmarks = frame['gt_landmark'].squeeze()
            estimated_landmarks = frame['predicted_landmark_3d'].squeeze()
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

