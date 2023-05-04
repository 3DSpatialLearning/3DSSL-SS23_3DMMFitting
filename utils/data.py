import os
import numpy as np
import torch
import cv2

def load_data(scale: float = 1.0, landmarks_path: str = None, points_path: str = None, normals_path: str = None, image_path: str = None):
    data = {}
    if landmarks_path is not None:
        data["landmarks"] = np.load(landmarks_path) * scale
    if points_path is not None:
        data["points"] = np.load(points_path) * scale
    if normals_path is not None:
        data["normals"] = np.load(normals_path)
    if image_path is not None:
        data["image"] = cv2.imread(image_path)
    return data

def load_camera_data(intrinsics_path: str = None, extrinsics_path: str = None):
    data = {}
    if intrinsics_path is not None:
        data["intrinsics"] = np.load(intrinsics_path)
    if extrinsics_path is not None:
        data["extrinsics"] = np.load(extrinsics_path)
    return data

"""
    Load data from a set of directories. If these are not None, then the data will be loaded from the corresponding files resulting 
    from the intersection of their file names.
    The returned dictionary has the following structure (if the corresponding directory is not None):
    {
        id_1: {
            "landmarks": np.array,
            "points": np.array,
            "normals": np.array,
            "image": np.array
        },
        id_2: {...}
"""

def resize_data_images(data: dict[str, dict], sx: float = 1.0, sy: float = 1.0) -> dict[str, dict]:
    frames = data['frames']
    for id, frame in frames.items():
        frames[id]['image'] = cv2.resize(frame['image'], (0, 0), fx=sx, fy=sy)
    data['frames'] = frames
    return data

def dict_tensor_to_np(data: dict[str, torch.tensor]) -> dict[str, np.array]:
    for k, v in data.items():
        data[k] = v.detach().cpu().squeeze().numpy()
    return data
