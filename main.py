import fire
import torch
import cv2
import numpy as np

from dataset.CameraFrameDataset import CameraFrameDataset
from dataset.transforms import ToTensor

from models.LandmarkDetector import DlibLandmarkDetector
from models.FaceReconstructionModel import FaceReconModel

from torch.utils.data import DataLoader
from torchvision.transforms.transforms import Compose as TransformCompose
from config import get_config
from utils.utils import check_size

"""
  Multi-camera multi-frame FLAME fitting pipeline.
  - data_dir: directory containing subfolders each correspoding to a camera data, with the following structure:
    data_dir
    ├── camera_1
    │   ├── colmap_consistency/
    │   ├── colmap_depth/
    │   ├── colmap_normals/
    │   ├── images/
    │   ├── intrinsics.npy
    │   └── extrinsics.npy
    ├── camera_2      ...
  - num_frames_for_shape_fitting (default: 1): number of frames to use for Flame shape fitting.
"""


def main(
        data_dir: str = "data/toy_task/multi_frame_rgbd_fitting",
):
    # data loading
    landmark_detector = DlibLandmarkDetector()
    transforms = TransformCompose([ToTensor()])
    dataset = CameraFrameDataset(data_dir, has_gt_landmarks=False, transform=transforms)
    dataset.precompute_landmarks(landmark_detector, force_precompute=False)
    dataloader = DataLoader(dataset, batch_size=dataset.num_cameras(), shuffle=False, num_workers=8)

    # get camera settings
    first_frame_features = next(iter(dataloader))

    extrinsic_matrices = first_frame_features["extrinsics"]
    intrinsic_matrices = first_frame_features["intrinsics"]

    # resolution = (first_frame_features["image"].shape[3] // 2, first_frame_features["image"].shape[2] // 2)
    # resolution = check_size(resolution, 8)
    resolution = (224, 224)
    aspect_ratio = resolution[1]/resolution[0]

    # get face reconstruction model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = get_config()

    face_recon_model = FaceReconModel(
        face_model_config=config,
        resolution=resolution,
        device=device,
    )

    face_recon_model.to(device)

    face_recon_model.set_transformation_matrices(
        extrinsic_matrices=extrinsic_matrices,
        intrinsic_matrices=intrinsic_matrices,
        aspect_ratio=aspect_ratio
    )

    for frame_num, frame_features in enumerate(dataloader):
        color, depth = face_recon_model.optimize(frame_features)
        depth = depth.squeeze().detach().cpu().numpy()
        color = color.squeeze().detach().cpu().numpy()
        print(np.max(depth))
        print(np.max(color))
        break


if __name__ == '__main__':
    fire.Fire(main)
