import numpy as np
import pyvista as pv
import torch
from Loss import DistanceLoss
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from flame.FLAME import FLAME
from flame.config import get_config

LANDMARKS_FILE = "./data/team1_landmarks.npy"
POINTS_FILE = "./data/team1_points.npy"
NORMALS_FILE = "./data/team1_normals.npy"

def visualize_points(landmarks: np.ndarray, points: np.ndarray):
    pv_landmarks = pv.PolyData(landmarks)
    pv_points = pv.PolyData(points)
    plotter = pv.Plotter()
    plotter.add_mesh(pv_landmarks, color='red', point_size=5)
    plotter.add_mesh(pv_points, color='green', opacity=0.1, point_size=1)
    plotter.show()

def visualize_fitted_flame(steps: int = 100, lr: float = 1e-3, wd: float = 1e-4):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    landmarks = np.load(LANDMARKS_FILE)
    landmarks = torch.from_numpy(landmarks).to(device)
    landmarks.unsqueeze(0)  # [1, 51, 3]
    points = torch.from_numpy(np.load(POINTS_FILE)).to(device)
    points.unsqueeze(0) # [1, N, 3]

    config = get_config()
    flame_model = FLAME(config)
    flame_model.to(device)
    shape = nn.Parameter(torch.zeros(1, 100).float().to(device))
    exp = nn.Parameter(torch.zeros(1, 50).float().to(device))
    pose = nn.Parameter(torch.zeros(1, 6).float().to(device))

    optimizer = torch.optim.Adam(
        [shape, exp, pose],
        lr=lr,
        weight_decay=wd
    )
    criterion = DistanceLoss()
    # Todo: add losses for vertices and a regularization term
    for step in range(steps):
        vertices, _, landmarks3d = flame_model(shape_params=shape, expression_params=exp, pose_params=pose)
        loss = criterion(landmarks, landmarks3d)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    shape = shape.detach().cpu().numpy().squeeze()
    exp = exp.detach().cpu().numpy().squeeze()
    pose = pose.detach().cpu().numpy().squeeze()
    print("pose:", pose)
    print("shape:", shape)
    print("exp:", exp)
    vertices = vertices.detach().cpu().numpy().squeeze()
    landmarks3d = landmarks3d.detach().cpu().numpy().squeeze()
    visualize_points(landmarks3d, vertices)

if __name__ == '__main__':
    # landmarks = np.load(LANDMARKS_FILE)
    # points = np.load(POINTS_FILE)
    # visualize_points(landmarks, points)
    visualize_fitted_flame()
