import torch
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import Compose as TransformCompose

from config import get_config
from dataset.CameraFrameDataset import CameraFrameDataset
from dataset.transforms import ToTensor
from models.LandmarkDetector import DlibLandmarkDetector
from flame.FLAME import FLAME
from utils.fitting import fit_flame_to_batched_frame_features

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
    ├── camera_2
      ...
  - num_frames_for_shape_fitting (default: 1): number of frames to use for Flame shape fitting.
"""

if __name__ == '__main__':

    config = get_config()
    print("Selected device:", config.device)

    # data loading
    landmark_detector = DlibLandmarkDetector()
    transforms = TransformCompose([ToTensor()])
    dataset = CameraFrameDataset(config.cam_data_dir, need_backprojection=True, has_gt_landmarks=True,
                                 transform=transforms)
    dataset.precompute_landmarks(landmark_detector, force_precompute=False)
    dataloader = DataLoader(dataset, batch_size=dataset.num_cameras(), shuffle=False, num_workers=0)

    # FLAME related variables
    flame_model = FLAME(config)
    flame_model.to(config.device)
    shape = torch.nn.Parameter(torch.zeros(1, config.shape_params).float().to(config.device))
    exp = torch.nn.Parameter(torch.zeros(1, config.expression_params).float().to(config.device))
    pose = torch.nn.Parameter(torch.zeros(1, config.pose_params).float().to(config.device))

    # fitting
    total_frame_count = len(dataloader)
    for frame_id, batch_features in enumerate(dataloader):
        print(f"Processing frame {frame_id}/{total_frame_count}")
        if frame_id == 0:
            exp.requires_grad = False
        else:
            exp.requires_grad = True
        if frame_id >= config.shape_fitting_frames:
            shape.requires_grad = False
        shape, exp, pose = fit_flame_to_batched_frame_features(
            frame_id,
            flame_model,
            shape,
            exp,
            pose,
            batch_features,
            config
        )