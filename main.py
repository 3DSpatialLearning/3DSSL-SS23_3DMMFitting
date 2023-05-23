import fire

from config import get_config

from dataset.CameraFrameDataset import CameraFrameDataset
from dataset.transforms import ToTensor

from models.LandmarkDetector import DlibLandmarkDetector
from models.FaceReconstructionModel import FaceReconModel

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
):
    # data loading
    landmark_detector = DlibLandmarkDetector()
    transforms = TransformCompose([ToTensor()])
    dataset = CameraFrameDataset(data_dir, has_gt_landmarks=False, transform=transforms)
    dataset.precompute_landmarks(landmark_detector, force_precompute=False)
    dataloader = DataLoader(dataset, batch_size=dataset.num_cameras(), shuffle=False, num_workers=0)

    # create the FaceRecon model
    face_recon_model_config = get_config()
    face_recon_model = FaceReconModel(
        face_model_config=face_recon_model_config,
        device="cuda"
    )

    # fitting
    for frame_num, frame_features in enumerate(dataloader):
        print(frame_features)
        pass


if __name__ == '__main__':
    fire.Fire(main)
