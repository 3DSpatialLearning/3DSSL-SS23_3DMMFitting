import torch
import cv2
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

from torch.utils.data import DataLoader
from torchvision.transforms.transforms import Compose as TransformCompose

from config import get_config

from dataset.CameraFrameDataset import CameraFrameDataset
from dataset.transforms import ToTensor

from models.FaceReconstructionModel import FaceReconModel
from models.HairSegmenter import HairSegmenter
from models.LandmarkDetectorPIPNET import LandmarkDetectorPIPENET

from pathlib import Path

if __name__ == '__main__':

    config = get_config()

    # data loading
    experiment_name = config.experiment_name
    output_frame_rate = config.output_frame_rate
    output_video_name = config.output_video_name
    workdir = config.workdir
    draw_landmarks = config.draw_landmarks
    deca_output_dir = Path(config.deca_output_path)

    deca_output_paths = sorted(deca_output_dir.glob("*.npy"))

    # Set project dir
    project_dir = (Path(workdir) / f"{experiment_name}").absolute()
    project_dir.mkdir(parents=True, exist_ok=True)
    run_dir = (
        project_dir
        / f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
    )
    run_dir.mkdir(parents=True)

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

    # Get the video writer
    height, width = face_recon_model.coarse2fine_resolutions[-1]
    video = cv2.VideoWriter(str(run_dir / output_video_name), cv2.VideoWriter_fourcc(*"mp4v"), output_frame_rate, (3 * width, height))

    # Get writer
    writer = SummaryWriter(log_dir=str(run_dir))

    # Get the scan to mesh distance list
    scan_to_mesh_distance_list = []
    scan_to_mesh_distance_deca_list = []

    for frame_num, (frame_features, deca_output) in enumerate(zip(dataloader, deca_output_paths)):
        print(f"Processing frame {frame_num} out of {len(dataset)}")
        landmark_mask = np.isin(frame_features["camera_id"], config.landmark_camera_id)
        color, _, input_color, _, flame_68_landmarks, flame_mp_landmarks, rgb_in_landmarks_mask, scan_to_mesh_distance, scan_to_mesh_distance_deca, _, _ = face_recon_model.optimize(frame_features, first_frame = frame_num == 0, deca_pred_verts=np.load(deca_output))

        # Log the losses
        writer.add_scalar('per frame scan to mesh distance', scan_to_mesh_distance.item(), frame_num)

        color = (color[rgb_in_landmarks_mask].squeeze(0).detach().cpu().numpy()[:, :, ::-1] * 255).astype(np.uint8)
        input_color = (input_color[rgb_in_landmarks_mask].squeeze(0).detach().cpu().contiguous().numpy()[:, :, ::-1] * 255).astype(np.uint8)

        if draw_landmarks:
            gt_landmarks = frame_features["predicted_landmark_2d"][landmark_mask].squeeze().detach().cpu().numpy()
            flame_68_landmarks = flame_68_landmarks[0].detach().cpu().numpy()
            flame_mp_landmarks = flame_mp_landmarks[0].detach().cpu().numpy()

            for gt_landmark, flame_landmark in zip(gt_landmarks, flame_68_landmarks):
                cv2.circle(color, (int(flame_landmark[0]), int(flame_landmark[1])), 2, (0, 0, 255), -1)
                cv2.circle(input_color, (int(gt_landmark[0] * input_color.shape[1] / first_frame_features["image"].shape[2:][1]), int(gt_landmark[1] * input_color.shape[0] / first_frame_features["image"].shape[2:][0])), 2, (0, 255, 0), -1)

            for flame_landmark in flame_mp_landmarks:
                cv2.circle(color, (int(flame_landmark[0]), int(flame_landmark[1])), 2, (255, 0, 0), -1)

        alpha = 0.5
        blended = (cv2.addWeighted(color, alpha, input_color, 1 - alpha, 0)).astype(np.uint8)

        combined_img = cv2.hconcat([blended, input_color, color])
        
        video.write(combined_img)

        scan_to_mesh_distance_list.append(scan_to_mesh_distance.item())
        scan_to_mesh_distance_deca_list.append(scan_to_mesh_distance_deca.item())

    # Compute the mean and std of the scan to mesh distance
    scan_to_mesh_distance_list = np.array(scan_to_mesh_distance_list)
    scan_to_mesh_distance_mean = np.mean(scan_to_mesh_distance_list)
    scan_to_mesh_distance_std = np.std(scan_to_mesh_distance_list)

    # Compute the mean and std of the scan to mesh distance for deca
    scan_to_mesh_distance_deca_list = np.array(scan_to_mesh_distance_deca_list)
    scan_to_mesh_distance_deca_mean = np.mean(scan_to_mesh_distance_deca_list)
    scan_to_mesh_distance_deca_std = np.std(scan_to_mesh_distance_deca_list)

    # Log the mean and std of the scan to mesh distance
    with open(str(run_dir / "scan_to_mesh_distance.txt"), "w") as f:
        f.write(f"Mean scan to mesh distance: {scan_to_mesh_distance_mean}\n")
        f.write(f"Std scan to mesh distance: {scan_to_mesh_distance_std}\n")

    # Log the mean and std of the scan to mesh distance for deca
    with open(str(run_dir / "scan_to_mesh_distance_deca.txt"), "w") as f:
        f.write(f"Mean scan to mesh distance: {scan_to_mesh_distance_deca_mean}\n")
        f.write(f"Std scan to mesh distance: {scan_to_mesh_distance_deca_std}\n")

    cv2.destroyAllWindows()
    video.release()