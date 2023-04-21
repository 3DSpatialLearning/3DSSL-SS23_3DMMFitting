import numpy as np
import torch
import torch.nn as nn
from pytorch3d.loss.chamfer import chamfer_distance
from flame.FLAME import FLAME
from flame.config import get_config
from utils.rigid_transform_3D import rigid_transform_3D
from utils.visualize import display_points

def fit_data(points: np.ndarray, landmarks: np.ndarray, normals: np.ndarray, steps: int = 100, lr: float = 0.01):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = get_config()
    flame_model = FLAME(config)
    flame_model.to(device)

    shape = nn.Parameter(torch.zeros(1, 100).float().to(device))
    exp = nn.Parameter(torch.zeros(1, 50).float().to(device))
    pose = nn.Parameter(torch.zeros(1, 6).float().to(device))

    # prepare model landmarks
    _, model_landmarks = flame_model(shape, exp, pose)
    model_landmarks = model_landmarks.squeeze().cpu().detach().numpy()

    mn = min(landmarks.shape[0], model_landmarks.shape[0]) - 1

    # 3d rigid transformation
    R, t = rigid_transform_3D(
        landmarks[:mn].transpose(),
        model_landmarks[:mn].transpose())
    landmarks = (R @ landmarks.transpose() + t).transpose()

    # optimization
    optimizer = torch.optim.Adam([shape, exp, pose], lr=lr)
    optimizer.zero_grad()

    landmarks = torch.tensor(landmarks).float().to(device).unsqueeze(0)
    for step in range(steps):
        _, model_landmarks = flame_model(shape_params=shape, expression_params=exp, pose_params=pose)
        loss, _ = chamfer_distance(landmarks, model_landmarks)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("step:",step,"/",steps,"loss(chamfer distance):",loss.item())
    
    return landmarks.squeeze().cpu().detach().numpy()