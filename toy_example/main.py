import torch
import torch.nn as nn
from flame.FLAME import FLAME
from flame.config import get_config
from pytorch3d.loss import chamfer_distance
from utils.visualization import visualize_points
from utils.data import dict_tensor_to_np, load_data

LANDMARKS_FILE = "./data/team1_landmarks.npy"
POINTS_FILE = "./data/team1_points.npy"
NORMALS_FILE = "./data/team1_normals.npy"

def visualize_points_to_fit():
    data = load_data(1/1000, LANDMARKS_FILE, POINTS_FILE, NORMALS_FILE)
    visualize_points(data)

def fitting(data:dict, steps: int = 100, lr: float = 1e-3, wd: float = 1e-4):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for k, v in data.items():
        data[k] = torch.from_numpy(v).float().unsqueeze(0).to(device)
    config = get_config()
    flame_model = FLAME(config)
    flame_model.to(device)
    shape = nn.Parameter(torch.zeros(1, 100).float().to(device))
    exp = nn.Parameter(torch.zeros(1, 50).float().to(device))
    pose = nn.Parameter(torch.zeros(1, 6).float().to(device))
    pose_optimizer = torch.optim.Adam(
        [pose],
        lr=lr,
        weight_decay=wd
    )
    shape_optimizer = torch.optim.Adam(
        [shape, exp],
        lr=lr,
        weight_decay=wd
    )
    # optimize for pose (alternatively use procrustes)
    pose.requires_grad = True
    exp.requires_grad = False
    shape.requires_grad = False
    for step in range(steps):
        _, landmarks3d = flame_model(shape_params=shape, expression_params=exp, pose_params=pose)
        loss, _ = chamfer_distance(data["landmarks"], landmarks3d)
        pose_optimizer.zero_grad()
        loss.backward()
        pose_optimizer.step()
    # optimize for shape, exp
    pose.requires_grad = False
    exp.requires_grad = True
    shape.requires_grad = True
    for step in range(steps):
        vertices, landmarks3d = flame_model(shape_params=shape, expression_params=exp, pose_params=pose)
        loss, _ = chamfer_distance(data["points"], vertices)
        shape_optimizer.zero_grad()
        loss.backward()
        shape_optimizer.step()
    dict_in_tensor = {
        "points": vertices,
        "landmarks": landmarks3d,
        "pose": pose,
        "shape": shape,
        "exp": exp
    }
    dict_in_np = dict_tensor_to_np(dict_in_tensor)
    return dict_in_np

def visualize_fitted_flame():
    data = load_data()
    fitted_data = fitting(data=data, steps=200)
    for k, v in fitted_data.items():
        fitted_data[k] = v.squeeze()
    print("pose:", fitted_data["pose"])
    print("shape:", fitted_data["shape"])
    print("exp:", fitted_data["exp"])
    visualize_points(fitted_data)

if __name__ == '__main__':
    visualize_points_to_fit()
    # visualize_fitted_flame()
