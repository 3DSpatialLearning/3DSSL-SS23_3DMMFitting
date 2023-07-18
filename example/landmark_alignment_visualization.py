import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import Compose as TransformCompose
from config import get_config

from dataset.CameraFrameDataset import CameraFrameDataset
from dataset.transforms import ToTensor
from models.LandmarkDetector import DlibLandmarkDetector
from flame.FLAME import FLAME
from utils.transform import rigid_transform_3d
from utils.visualization import visualize_landmark_alignment

if __name__ == '__main__':
    config = get_config(path_to_data_dir="../")
    # data loading
    landmark_detector = DlibLandmarkDetector(path_to_dlib_predictor_model="../data/checkpoints/shape_predictor_68_face_landmarks.dat")
    transforms = TransformCompose([ToTensor()])
    dataset = CameraFrameDataset(config.cam_data_dir, need_backprojection=True, has_gt_landmarks=True,
                                 transform=transforms)
    dataset.precompute_landmarks(landmark_detector, force_precompute=False)
    dataloader = DataLoader(dataset, batch_size=dataset.num_cameras(), shuffle=False, num_workers=0)

    # FLAME related variables
    flame_model = FLAME(config)
    flame_model.to(config.device)
    shape = torch.nn.Parameter(torch.zeros(1, config.shape_params, dtype=torch.float32).to(config.device))
    exp = torch.nn.Parameter(torch.zeros(1, config.expression_params, dtype=torch.float32).to(config.device))
    pose = torch.nn.Parameter(torch.zeros(1, config.pose_params, dtype=torch.float32).to(config.device))
    shape.requires_grad = False
    exp.requires_grad = False
    pose.requires_grad = False

    # fitting
    total_frame_count = len(dataloader)
    for frame_id, batch_features in enumerate(dataloader):
        print(f"Frame {frame_id}/{total_frame_count}")
        flame_model = flame_model.to(config.device)
        flame_model_faces = torch.from_numpy(flame_model.faces.astype(np.int32)).unsqueeze(0).to(config.device)
        shape = shape.to(config.device)
        exp = exp.to(config.device)
        pose = pose.to(config.device)

        _, flame_landmarks_before_alignment = flame_model(shape, exp, pose)

        # get alignment from flame to input with Procrustes and set flame pose parameter
        landmarks_input = batch_features['gt_landmark'].squeeze().cpu().detach().numpy()
        landmarks_not_nan_indices = ~(np.isnan(landmarks_input).any(axis=1))
        landmarks_input = landmarks_input[landmarks_not_nan_indices]
        flame_landmarks_before_alignment = flame_landmarks_before_alignment.squeeze().cpu().detach().numpy()[landmarks_not_nan_indices].transpose()
        landmarks_input_transposed = landmarks_input.transpose()

        r, t = rigid_transform_3d(flame_landmarks_before_alignment, landmarks_input_transposed)
        rot = torch.from_numpy(r).to(config.device)
        transl = torch.from_numpy(t.T).to(config.device)

        _, flame_landmarks_after_alignment = flame_model(shape_params=shape, expression_params=exp,
                                                              pose_params=pose, rot=rot, transl=transl)
        flame_landmarks_after_alignment = flame_landmarks_after_alignment.squeeze().cpu().detach().numpy()
        flame_landmarks_after_alignment = flame_landmarks_after_alignment[landmarks_not_nan_indices]

        # visualize aligned landmarks
        visualize_landmark_alignment(landmarks_input, flame_landmarks_before_alignment.transpose(), flame_landmarks_after_alignment)