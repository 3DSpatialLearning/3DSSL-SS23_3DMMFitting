import fire
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

def main(
        cam_data_dir: str = "data/toy_task/multi_frame_rgbd_fitting",
        num_frames_for_shape_fitting: int = 3,
        device: str = None,
):
    config = get_config()

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # data loading
    landmark_detector = DlibLandmarkDetector()
    transforms = TransformCompose([ToTensor()])
    dataset = CameraFrameDataset(cam_data_dir, need_backprojection=True, has_gt_landmarks=True, transform=transforms)
    dataset.precompute_landmarks(landmark_detector, force_precompute=False)
    dataloader = DataLoader(dataset, batch_size=dataset.num_cameras(), shuffle=False, num_workers=0)

    # FLAME related variables
    flame_model = FLAME(config)
    flame_model.to(device)
    shape = torch.nn.Parameter(torch.zeros(1, config.shape_params).float().to(device))
    exp = torch.nn.Parameter(torch.zeros(1, config.expression_params).float().to(device))
    pose = torch.nn.Parameter(torch.zeros(1, config.pose_params).float().to(device))

    # fitting
    for frame, batch_features in enumerate(dataloader):
        print(f"Processing frame {frame}")
        if frame >= num_frames_for_shape_fitting:
            shape.requires_grad = False
        shape, exp, pose = fit_flame_to_batched_frame_features(
            frame,
            flame_model,
            shape,
            exp,
            pose,
            batch_features,
            device,
            config
        )

if __name__ == '__main__':
  fire.Fire(main)