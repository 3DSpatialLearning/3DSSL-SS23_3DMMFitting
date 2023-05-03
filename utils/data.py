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
def load_batch_data(scale: float = 1.0, landmarks_dir: str = None, points_dir: str = None, normals_dir: str = None, image_dir: str = None) \
        -> dict[str, dict]:
    data = {}
    if landmarks_dir is not None:
        landmark_id_to_path = {}
        files = os.listdir(landmarks_dir)
        for file in files:
            if ".npy" in file:
                file_path = os.path.join(landmarks_dir, file)
                id = os.path.splitext(file)[0]
                landmark_id_to_path[id] = file_path
    if points_dir is not None:
        point_id_to_path = {}
        files = os.listdir(points_dir)
        for file in files:
            if ".npy" in file:
                file_path = os.path.join(points_dir, file)
                id = os.path.splitext(file)[0]
                point_id_to_path[id] = file_path
    if normals_dir is not None:
        normal_id_to_path = {}
        files = os.listdir(normals_dir)
        for file in files:
            if ".npy" in file:
                file_path = os.path.join(normals_dir, file)
                id = os.path.splitext(file)[0]
                normal_id_to_path[id] = file_path
    if image_dir is not None:
        image_id_to_path = {}
        files = os.listdir(image_dir)
        for file in files:
            if ".png" in file:
                file_path = os.path.join(image_dir, file)
                id = os.path.splitext(file)[0]
                image_id_to_path[id] = file_path

    list_keys = []
    if landmark_id_to_path is not None:
        list_keys.append(landmark_id_to_path.keys())
    if point_id_to_path is not None:
        list_keys.append(point_id_to_path.keys())
    if normal_id_to_path is not None:
        list_keys.append(normal_id_to_path.keys())
    if image_id_to_path is not None:
        list_keys.append(image_id_to_path.keys())
    valid_ids = list(set.intersection(*map(set, list_keys)))
    for id in valid_ids:
        data[id] = load_data(scale=scale,
                            landmarks_path=landmark_id_to_path[id] if landmark_id_to_path is not None else None,
                            points_path=point_id_to_path[id] if point_id_to_path is not None else None,
                            normals_path=normal_id_to_path[id] if normal_id_to_path is not None else None,
                            image_path=image_id_to_path[id] if image_id_to_path is not None else None)
    return data

def resize_data_images(data: dict[str, dict], sx: float = 1.0, sy: float = 1.0) -> dict[str, dict]:
    frames = data['frames']
    for id, frame in frames.items():
        frames[id]['image'] = cv2.resize(frame['image'], (0, 0), fx=sx, fy=sy)
    data['frames'] = frames
    return data

def dict_tensor_to_np(data: dict[str, torch.tensor]):
    for k, v in data.items():
        data[k] = v.detach().cpu().numpy()
