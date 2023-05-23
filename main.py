import fire
from dataset.CameraFrameDataset import CameraFrameDataset
from dataset.transforms import ToTensor
from models.LandmarkDetector import DlibLandmarkDetector
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import Compose as TransformCompose
from flame.config import get_config as flame_config

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
       data_dir: str = "data/toy_task/multi_frame_rgbd_fitting",
       num_frames_for_shape_fitting: int = 1,
):
  # data loading
  landmark_detector = DlibLandmarkDetector()
  transforms = TransformCompose([ToTensor()])
  dataset = CameraFrameDataset(data_dir, has_gt_landmarks=False, transform=transforms)
  dataset.precompute_landmarks(landmark_detector, force_precompute=False)
  dataloader = DataLoader(dataset, batch_size=dataset.num_cameras(), shuffle=False, num_workers=0)


  # fitting
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  shape = torch.nn.Parameter(torch.zeros(1, flame_config.shape_params).float().to(device))
  exp = torch.nn.Parameter(torch.zeros(1, flame_config.expression_params).float().to(device))
  pose = torch.nn.Parameter(torch.zeros(1, flame_config.pose_params).float().to(device))
  for frame, frame_batch in enumerate(dataloader):
    print(f"Processing frame {frame}")

if __name__ == '__main__':
  fire.Fire(main)