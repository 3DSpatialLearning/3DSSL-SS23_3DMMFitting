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

def load_data():
    landmarks = np.load(LANDMARKS_FILE)
    points = np.load(POINTS_FILE)
    normals = np.load(NORMALS_FILE)
    data = {
        'landmarks': landmarks,
        "points": points,
        "normals": normals
    }
    return data

def visualize_points(data: dict[str, np.ndarray]):
    pv_landmarks = pv.PolyData(data["landmarks"])
    pv_points = pv.PolyData(data["points"])
    plotter = pv.Plotter()
    plotter.add_mesh(pv_landmarks, color='red', point_size=5)
    plotter.add_mesh(pv_points, color='green', opacity=0.1, point_size=1)
    plotter.show()

def visualize_points_to_fit():
    data = load_data()
    visualize_points(data)

def fitting(data:dict, steps: int = 100, lr: float = 1e-3, wd: float = 1e-4):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for k, v in data.items():
        data[k] = torch.from_numpy(v).unsqueeze(0).to(device)
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
        loss = criterion(data["landmarks"], landmarks3d)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return shape, exp, pose, vertices, landmarks3d

def visualize_fitted_flame():
    data = load_data()
    shape, exp, pose, out_points, out_landmarks = fitting(data=data, steps=200)
    shape = shape.detach().cpu().numpy().squeeze()
    exp = exp.detach().cpu().numpy().squeeze()
    pose = pose.detach().cpu().numpy().squeeze()
    print("pose:", pose)
    print("shape:", shape)
    print("exp:", exp)
    out_data = {
        "landmarks": out_landmarks.detach().cpu().numpy().squeeze(),
        "points": out_points.detach().cpu().numpy().squeeze()
    }
    visualize_points(out_data)

if __name__ == '__main__':
    visualize_points_to_fit()
    # visualize_fitted_flame()
