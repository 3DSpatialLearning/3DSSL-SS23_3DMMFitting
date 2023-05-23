import fire
from dataset.CameraFrameDataset import CameraFrameDataset
from dataset.transforms import ToTensor
from models.LandmarkDetector import DlibLandmarkDetector
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import Compose as TransformCompose

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
  for frame, frame_batch in enumerate(dataloader):
    pass

if __name__ == '__main__':
  fire.Fire(main)