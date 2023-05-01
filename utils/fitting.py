import numpy as np
import torch
import torch.nn as nn
from flame.FLAME import FLAME
from flame.config import get_config
from utils.transform import rigid_transform_3d
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from utils.loss import scan_to_mesh_distance
from utils.visualization import visualize_3d_scan_and_3d_face_model
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
import random
from torch.utils.tensorboard import SummaryWriter


def fit_flame_model_to_input_point_cloud(
        input_data: dict[str, np.ndarray],
        steps: int,
        lr: float,
        display_results: bool = True,
        scan_to_mesh_weight: int = 2,
        landmark_weight: int = 1e-2,
        shape_weight: float = 1e-8,
        exp_weight: float = 1e-9,
):
    writer = SummaryWriter()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = get_config()
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
        lr=lr,
    )
    scheduler = ExponentialLR(optimizer, gamma=0.999)

    scan_to_mesh_dist = scan_to_mesh_distance
    landmark_dist = nn.MSELoss()

    pbar = tqdm(range(steps))

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

        loss = scan_to_mesh_weight * scan_to_mesh_loss + landmark_weight * landmark_loss + \
               shape_weight * shape.norm(dim=1, p=2) + exp_weight * exp.norm(dim=1, p=2)

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
