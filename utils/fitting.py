import numpy as np
import torch
import torch.nn as nn
import random
import argparse
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

from torch.utils.tensorboard import SummaryWriter
from flame.FLAME import FLAME
from utils.transform import rigid_transform_3d, rotation_matrix_to_axis_angle
from utils.loss import scan_to_mesh_distance, scan_to_mesh_face_distance
from utils.visualization import visualize_3d_scan_and_3d_face_model
from dataset.utils import to_device

from pytorch3d.loss import chamfer_distance

"""
    Deprecated: only used in the toy task
"""
def fit_flame_to_batched_frame_features(
        frame_id: int,
        flame_model: FLAME,
        shape: nn.Parameter,
        exp: nn.Parameter,
        pose: nn.Parameter,
        frame_batch: dict[str, torch.tensor],
        config: argparse.Namespace
):
    flame_model = flame_model.to(config.device)
    flame_model_faces = torch.from_numpy(flame_model.faces.astype(np.int32)).unsqueeze(0).to(config.device)
    shape = shape.to(config.device)
    exp = exp.to(config.device)
    pose = pose.to(config.device)

    _, landmarks_flame = flame_model(shape, exp, pose)

    # get alignment from flame to input with Procrustes and set flame pose parameter
    landmarks_input = frame_batch['gt_landmark'].squeeze().cpu().detach().numpy()
    landmarks_not_nan_indices = ~(np.isnan(landmarks_input).any(axis=1))
    landmarks_input = landmarks_input[landmarks_not_nan_indices]
    landmarks_flame = landmarks_flame.squeeze().cpu().detach().numpy()[landmarks_not_nan_indices].transpose()
    landmarks_input_transposed = landmarks_input.transpose()
    r, t = rigid_transform_3d(landmarks_flame, landmarks_input_transposed)
    rot = torch.from_numpy(r).to(config.device)
    transl = torch.from_numpy(t.T).to(config.device)

    # optimize flame shape and expression
    frame_batch = to_device(frame_batch, config.device)
    optimizer = torch.optim.Adam(
        [{'params': [shape, exp, pose]}],
        lr=config.lr,
    )
    scheduler = ExponentialLR(optimizer, gamma=0.999)
    landmark_dist = nn.MSELoss()
    pbar = tqdm(range(config.steps))
    for _ in pbar:
        optimizer.zero_grad()
        predicted_vertices, predicted_landmarks = flame_model(shape_params=shape, expression_params=exp,
                                                              pose_params=pose, rot=rot, transl=transl)

        mesh = Meshes(verts=predicted_vertices, faces=flame_model_faces)
        random_indices = random.sample(list(range(frame_batch['point'].shape[1])), config.num_samples)
        random_points = frame_batch['point'][:, random_indices]
        random_normals = frame_batch['point_normal'][:, random_indices]
        scan_to_mesh_loss, _ = scan_to_mesh_distance(random_points, random_normals,
                                                 *sample_points_from_meshes(mesh, num_samples=config.num_samples, return_normals=True))
        scan_to_face_loss = scan_to_mesh_face_distance(random_points, mesh)
        landmark_loss = landmark_dist(frame_batch['gt_landmark'][:, landmarks_not_nan_indices], predicted_landmarks[:, landmarks_not_nan_indices])
        loss = config.scan_to_mesh_weight * scan_to_mesh_loss + config.scan_to_face_weight * scan_to_face_loss + config.landmark_weight * landmark_loss + \
               config.shape_regularization_weight * shape.norm(dim=1, p=2) + config.exp_regularization_weight * exp.norm(dim=1, p=2)
        loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.set_postfix(loss=loss.item())

    visualize_3d_scan_and_3d_face_model(
        points_scan=frame_batch['point'].squeeze().cpu().numpy(),
        points_3d_face=predicted_vertices.detach().cpu().squeeze().numpy(),
        faces_3d_face=flame_model.faces,
        predicted_landmarks_3d=landmarks_input,
        # screenshot_path=f"{frame_id}.png",
    )

    return shape, exp, pose


"""
    Deprecated: only used in the toy task
"""
def fit_flame_model_to_input_point_cloud(
        input_data: dict[str, np.ndarray],
        config: argparse.Namespace,
        display_results: bool = True,
):
    writer = SummaryWriter()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    flame_model = FLAME(config)
    flame_model_faces = torch.from_numpy(flame_model.faces.astype(np.int32)).unsqueeze(0).to(device)
    flame_model.to(device)

    shape = nn.Parameter(torch.zeros(1, config.shape_params).float().to(device))
    exp = nn.Parameter(torch.zeros(1, config.expression_params).float().to(device))
    pose = nn.Parameter(torch.zeros(1, config.pose_params).float().to(device))

    # Compute the optimal rigid transform that aligns the input 3d point cloud to Flame model

    flame_vertices, flame_landmarks = flame_model(shape, exp, pose)
    flame_landmarks = flame_landmarks.squeeze().cpu().detach().numpy()
    flame_vertices = flame_vertices.squeeze().cpu().detach().numpy()

    # test
    not_nan_indices = ~(np.isnan(input_data['landmarks']).any(axis=1))
    landmarks_source = flame_landmarks[not_nan_indices].transpose()
    landmarks_input = input_data['landmarks'][not_nan_indices].transpose()

    r, t = rigid_transform_3d(landmarks_input, landmarks_source)

    input_landmarks = torch.tensor((r @ input_data['landmarks'].transpose() + t).transpose()).float().to(
        device).unsqueeze(0)
    input_points = torch.tensor((r @ input_data['points'].transpose() + t).transpose()).float().to(device).unsqueeze(0)
    input_normals = torch.tensor((r @ input_data['normals'].transpose()).transpose()).float().to(device).unsqueeze(0)

    if display_results:
        print("Displaying face model after coarse alignment...")
        visualize_3d_scan_and_3d_face_model(
            points_scan=input_points.squeeze().cpu().numpy(),
            points_3d_face=flame_vertices,
            faces_3d_face=flame_model.faces
        )

    optimizer = torch.optim.Adam(
        [{'params': [shape, exp, pose]}],
        lr=config.lr,
    )
    scheduler = ExponentialLR(optimizer, gamma=0.999)

    scan_to_mesh_dist = scan_to_mesh_distance
    landmark_dist = nn.MSELoss()

    pbar = tqdm(range(config.steps))

    num_samples = 20000
    for i in pbar:
        optimizer.zero_grad()
        predicted_vertices, predicted_landmarks = flame_model(shape_params=shape, expression_params=exp,
                                                              pose_params=pose)
        mesh = Meshes(verts=predicted_vertices, faces=flame_model_faces)

        random_indices = random.sample(list(range(input_points.shape[1])), num_samples)
        scan_to_mesh_loss, _ = scan_to_mesh_dist(input_points[:, random_indices], input_normals[:, random_indices],
                                                 *sample_points_from_meshes(mesh, num_samples=num_samples,
                                                                            return_normals=True))
        landmark_loss = landmark_dist(input_landmarks[:, not_nan_indices], predicted_landmarks[:, not_nan_indices])

        loss = config.scan_to_mesh_weight * scan_to_mesh_loss + config.landmark_weight * landmark_loss + \
               config.shape_regularization_weight * shape.norm(dim=1, p=2) + config.exp_regularization_weight * exp.norm(dim=1, p=2)

        writer.add_scalar('Loss/scan_to_mesh', scan_to_mesh_loss.item(), i)
        writer.add_scalar('Loss/landmark_loss', landmark_loss, i)

        loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.set_postfix(loss=loss.item())

    optimal_parameters = {
        "pose": pose.detach().cpu().squeeze().numpy(),
        "shape": shape.detach().cpu().squeeze().numpy(),
        "exp": exp.detach().cpu().squeeze().numpy()
    }

    if display_results:
        print("Displaying fitted face model")
        visualize_3d_scan_and_3d_face_model(
            points_scan=input_points.squeeze().cpu().numpy(),
            points_3d_face=predicted_vertices.detach().cpu().squeeze().numpy(),
            faces_3d_face=flame_model.faces
        )
    return optimal_parameters
