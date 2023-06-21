import torch
import cv2
import numpy as np

from torch.utils.data import DataLoader
from torchvision.transforms.transforms import Compose as TransformCompose

from config import get_config

from dataset.CameraFrameDataset import CameraFrameDataset
from dataset.transforms import ToTensor

from models.FaceReconstructionModel import FaceReconModel
from models.HairSegmenter import HairSegmenter
from models.LandmarkDetectorPIPNET import LandmarkDetectorPIPENET

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

    # data loading
    landmark_detector = LandmarkDetectorPIPENET()
    hair_segmentor = HairSegmenter()

    transforms = TransformCompose([ToTensor()])
    dataset = CameraFrameDataset(config.cam_data_dir, need_backprojection=True, has_gt_landmarks=True, transform=transforms)
    dataset.precompute_landmarks(landmark_detector, force_precompute=False)
    dataset.precompute_hair_masks(hair_segmentor, force_precompute=False)

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
        color, depth, input_color, input_depth, flame_68_landmarks, flame_mp_landmarks = face_recon_model.optimize(frame_features, first_frame = frame_num == 0)

        color = (color[0].detach().cpu().numpy()[:, :, ::-1] * 255).astype(np.uint8)
        depth = depth[0].detach().cpu().numpy()
        input_color = (input_color[0].detach().cpu().contiguous().numpy()[:, :, ::-1] * 255).astype(np.uint8)
        input_depth = input_depth[0].detach().cpu().numpy()

        gt_landmarks = frame_features["predicted_landmark_2d"][0].detach().cpu().numpy()
        flame_68_landmarks = flame_68_landmarks[0].detach().cpu().numpy()
        flame_mp_landmarks = flame_mp_landmarks[0].detach().cpu().numpy()

        for gt_landmark, flame_landmark in zip(gt_landmarks, flame_68_landmarks):
            cv2.circle(color, (int(flame_landmark[0]), int(flame_landmark[1])), 2, (0, 0, 255), -1)
            cv2.circle(input_color, (int(gt_landmark[0] * input_color.shape[1] / first_frame_features["image"].shape[2:][1]), int(gt_landmark[1] * input_color.shape[0] / first_frame_features["image"].shape[2:][0])), 2, (0, 255, 0), -1)

        for flame_landmark in flame_mp_landmarks:
            cv2.circle(color, (int(flame_landmark[0]), int(flame_landmark[1])), 2, (255, 0, 0), -1)

        alpha = 0.6
        blended = (cv2.addWeighted(color, alpha, input_color, 1 - alpha, 0)).astype(np.uint8)


        cv2.imwrite(f"./output/blended_{frame_num}.png", blended)
        cv2.imwrite(f"./output/input_{frame_num}.png", input_color)
        cv2.imwrite(f"./output/rendered_{frame_num}.png", color)

        cv2.imshow("blended", blended)
        cv2.imshow("original", input_color)
        cv2.imshow("rendered", color)
        cv2.waitKey(0)


