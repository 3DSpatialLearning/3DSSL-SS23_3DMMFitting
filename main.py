import fire
import torch
import cv2

from torch.utils.data import DataLoader
from torchvision.transforms.transforms import Compose as TransformCompose

from config import get_config

from dataset.CameraFrameDataset import CameraFrameDataset
from dataset.transforms import ToTensor

from models.LandmarkDetector import DlibLandmarkDetector
from models.FaceReconstructionModel import FaceReconModel

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

torch.backends.cudnn.benchmark = True


def main(
        cam_data_dir: str = "data/toy_task/multi_frame_rgbd_fitting",
        num_frames_for_shape_fitting: int = 3,
):
    config = get_config()

    # data loading
    landmark_detector = DlibLandmarkDetector()
    transforms = TransformCompose([ToTensor()])
    dataset = CameraFrameDataset(cam_data_dir, need_backprojection=True, has_gt_landmarks=True, transform=transforms)
    dataset.precompute_landmarks(landmark_detector, force_precompute=False)
    dataloader = DataLoader(dataset, batch_size=dataset.num_cameras(), shuffle=False, num_workers=0)

    # get camera settings
    first_frame_features = next(iter(dataloader))

    extrinsic_matrices = first_frame_features["extrinsics"]
    intrinsic_matrices = first_frame_features["intrinsics"]

    # get face reconstruction model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    face_recon_model = FaceReconModel(
        face_model_config=config,
        orig_img_shape=first_frame_features["image"].shape[2:],
        device=device,
    )
    face_recon_model.to(device)

    # set transformation matrices for projection
    face_recon_model.set_transformation_matrices(
        extrinsic_matrices=extrinsic_matrices,
        intrinsic_matrices=intrinsic_matrices,
    )

    # compute the initial alignment
    gt_landmark = first_frame_features["predicted_landmark_3d"][0]
    face_recon_model.set_initial_pose(gt_landmark)

    for frame_num, frame_features in enumerate(dataloader):
        color, depth = face_recon_model.optimize(frame_features)
        depth = depth[0].detach().cpu().numpy()
        color = color[0].detach().cpu().numpy()
        cv2.imshow("img", color.transpose(1, 2, 0)[:,:,::-1])
        cv2.waitKey(0)
        break


if __name__ == '__main__':
    fire.Fire(main)
