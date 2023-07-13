import torch
import cv2
import numpy as np

from torch.utils.data import DataLoader
from torchvision.transforms.transforms import Compose as TransformCompose

from config import get_config

from dataset.CameraFrameDataset import CameraFrameDataset
from dataset.transforms import ToTensor

from models_.FaceReconstructionModel import FaceReconModel
from models_.HairSegmenter import HairSegmenter
from models_.LandmarkDetectorPIPNET import LandmarkDetectorPIPENET

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

    rgb_camera_ids = config.rgb_camera_ids
    rgb_camera_mask = np.isin(first_frame_features["camera_id"], rgb_camera_ids)
    extrinsic_matrices_optimization = first_frame_features["extrinsics"][rgb_camera_mask]
    intrinsic_matrices_optimization = first_frame_features["intrinsics"][rgb_camera_mask]

    mesh_2_scan_camera_ids = config.scan_2_mesh_camera_ids
    mesh_2_scan_camera_mask = np.isin(first_frame_features["camera_id"], mesh_2_scan_camera_ids)
    extrinsic_matrices_fused_point_cloud = first_frame_features["extrinsics"][mesh_2_scan_camera_mask]

    # get face reconstruction model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    face_recon_model = FaceReconModel(
        face_model_config=config,
        orig_img_shape=first_frame_features["image"].shape[2:],
        device=device,
    )
    face_recon_model.to(device)

    # set transformation matrices for projection
    face_recon_model.set_transformation_matrices_for_optimization(
        extrinsic_matrices=extrinsic_matrices_optimization,
        intrinsic_matrices=intrinsic_matrices_optimization,
    )

    face_recon_model.set_transformation_matrices_for_fused_point_cloud(
        extrinsic_matrices=extrinsic_matrices_fused_point_cloud,
    )
    
    # compute the initial alignment
    face_recon_model.set_initial_pose(first_frame_features)

    for frame_num, frame_features in enumerate(dataloader):
        landmark_mask = np.isin(frame_features["camera_id"], config.landmark_camera_id)
        color, depth, input_color, input_depth, flame_68_landmarks, flame_mp_landmarks, rgb_in_landmarks_mask, scan_to_mesh_distance, _ = face_recon_model.optimize(frame_features, first_frame = frame_num == 0)

        
        color = (color[rgb_in_landmarks_mask].squeeze(0).detach().cpu().numpy()[:, :, ::-1] * 255).astype(np.uint8)
        input_color = (input_color[rgb_in_landmarks_mask].squeeze(0).detach().cpu().contiguous().numpy()[:, :, ::-1] * 255).astype(np.uint8)

        gt_landmarks = frame_features["predicted_landmark_2d"][landmark_mask].squeeze().detach().cpu().numpy()
        flame_68_landmarks = flame_68_landmarks[0].detach().cpu().numpy()
        flame_mp_landmarks = flame_mp_landmarks[0].detach().cpu().numpy()

        for gt_landmark, flame_landmark in zip(gt_landmarks, flame_68_landmarks):
            cv2.circle(color, (int(flame_landmark[0]), int(flame_landmark[1])), 2, (0, 0, 255), -1)
            cv2.circle(input_color, (int(gt_landmark[0] * input_color.shape[1] / first_frame_features["image"].shape[2:][1]), int(gt_landmark[1] * input_color.shape[0] / first_frame_features["image"].shape[2:][0])), 2, (0, 255, 0), -1)

        for flame_landmark in flame_mp_landmarks:
            cv2.circle(color, (int(flame_landmark[0]), int(flame_landmark[1])), 2, (255, 0, 0), -1)

        alpha = 0.6
        blended = (cv2.addWeighted(color, alpha, input_color, 1 - alpha, 0)).astype(np.uint8)


        # cv2.imwrite(f"./output/blended_{frame_num}.png", blended)
        # cv2.imwrite(f"./output/input_{frame_num}.png", input_color)
        # cv2.imwrite(f"./output/rendered_{frame_num}.png", color)

        cv2.imwrite(f"./blended_{frame_num}.png", blended)
        cv2.imwrite(f"./input_{frame_num}.png", input_color)
        cv2.imwrite(f"./rendered_{frame_num}.png", color)

        # cv2.imshow("blended", blended)
        # cv2.imshow("original", input_color)
        # cv2.imshow("rendered", color)
        # cv2.waitKey(0)


