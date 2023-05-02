import numpy as np
import torch

def load_data(scale: float = 1.0, landmarks_path: str = None, points_path: str = None, normals_path: str = None):
    data = {}
    if landmarks_path is not None:
        data["landmarks"] = np.load(landmarks_path) * scale
    if points_path is not None:
        data["points"] = np.load(points_path) * scale
    if normals_path is not None:
        data["normals"] = np.load(normals_path)
    return data

def dict_tensor_to_np(data: dict[str, torch.tensor]):
    for k, v in data.items():
        data[k] = v.detach().cpu().numpy()
