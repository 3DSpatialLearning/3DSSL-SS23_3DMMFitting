import numpy as np
import torch
import torch.nn as nn
from flame.FLAME import FLAME
from flame.config import get_config
from utils.transform import rigid_transform_3d
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from utils.loss import custom_chamfer_distance_single_direction
from utils.visualization import visualize_3d_scan_and_3d_face_model


def fit_flame_model_to_input_point_cloud(
        input_data: dict[str, np.ndarray],
        steps: int,
        lr: float,
        wd: float = 1e-4,
        display_results: bool = True,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = get_config()
    flame_model = FLAME(config)
    flame_model.to(device)

    shape = nn.Parameter(torch.zeros(1, config.shape_params).float().to(device))
    exp = nn.Parameter(torch.zeros(1, config.expression_params).float().to(device))
    pose = nn.Parameter(torch.zeros(1, config.pose_params).float().to(device))

    # Compute the optimal rigid transform that aligns the input 3d point cloud to Flame model

    flame_vertices, flame_landmarks = flame_model(shape, exp, pose)
    flame_landmarks = flame_landmarks.squeeze().cpu().detach().numpy()
    flame_vertices = flame_vertices.squeeze().cpu().detach().numpy()

    # Ask Simon to get the complete set of landmarks
    landmarks_input = input_data['landmarks'][15:24].transpose()
    landmarks_source = flame_landmarks[27:36].transpose()

    r, t = rigid_transform_3d(landmarks_input, landmarks_source)

    input_landmarks = torch.tensor((r @ input_data['landmarks'].transpose() + t).transpose()).float().to(device).unsqueeze(0)
    input_points = torch.tensor((r @ input_data['points'].transpose() + t).transpose()).float().to(device).unsqueeze(0)

    if display_results:
        print("Displaying face model after coarse alignment...")
        visualize_3d_scan_and_3d_face_model(
            points_scan=input_points.squeeze().cpu().numpy(),
            points_3d_face=flame_vertices,
            faces_3d_face=flame_model.faces
        )

    optimizer = torch.optim.Adam(
        [{'params': [shape, exp], 'weight_decay': wd},
         {'params': [pose], 'weight_decay': 0.0}],
        lr=lr,
    )
    scheduler = ExponentialLR(optimizer, gamma=0.999)
    loss_fn = custom_chamfer_distance_single_direction

    pbar = tqdm(range(steps))

    for _ in pbar:
        optimizer.zero_grad()
        predicted_vertices, predicted_landmarks = flame_model(shape_params=shape, expression_params=exp, pose_params=pose)
        loss, _ = loss_fn(predicted_vertices, input_points)
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
