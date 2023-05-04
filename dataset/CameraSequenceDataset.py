import numpy as np
from torch.utils.data import Dataset
import os
import cv2

"""
    This dataset is used to load a all relevant data related to a sequence of frames
    coming from a camara.
    It expects the following directory structure:
    - path_to_cam_dir
        - intrinsics.npy
        - extrinsics.npy
        - colmap_consistency
            - {id}.npy
            ...
        - colmap_depth
            - {id}.npy
            ...
        - colmap_normals
            - {id}.npy
            ..
        - images
            - {id}.png
            ...
        - landmarks
            - {id}.npy
            ...
"""
class CameraSequenceDataset(Dataset):

    def _compute_consistent_frame_ids(self):
        consistency_files = os.listdir(self.path_to_consistency_dir)
        consistency_files = [os.path.splitext(file)[0] for file in consistency_files if ".npy" in file]
        frame_ids = consistency_files # initialize the resulting list with consistency files
        depth_files = os.listdir(self.path_to_depth_dir)
        depth_files = [os.path.splitext(file)[0] for file in depth_files if ".npy" in file]
        frame_ids = list(set(frame_ids).intersection(depth_files))
        normal_files = os.listdir(self.path_to_normal_dir)
        normal_files = [os.path.splitext(file)[0] for file in normal_files if ".npy" in file]
        frame_ids = list(set(frame_ids).intersection(normal_files))
        image_files = os.listdir(self.path_to_image_dir)
        image_files = [os.path.splitext(file)[0] for file in image_files if ".png" in file]
        frame_ids = list(set(frame_ids).intersection(image_files))
        if self.has_gt_landmarks:
            landmark_files = os.listdir(self.path_to_landmarks_dir)
            landmark_files = [os.path.splitext(file)[0] for file in landmark_files if ".npy" in file]
            frame_ids = list(set(frame_ids).intersection(landmark_files))

        self.frame_ids = sorted(frame_ids)

    def __init__(self, path_to_cam_dir: str, has_gt_landmarks: bool = False):
        self.has_gt_landmarks = has_gt_landmarks
        assert os.path.exists(path_to_cam_dir), f"{path_to_cam_dir} directory does not exist!"
        # load camera data
        self.camera_intrinsics = np.load(os.path.join(path_to_cam_dir, "intrinsics.npy"))
        self.camera_extrinsics = np.load(os.path.join(path_to_cam_dir, "extrinsics.npy"))
        # store paths to sequence data
        self.path_to_consistency_dir = os.path.join(path_to_cam_dir, "colmap_consistency")
        self.path_to_depth_dir = os.path.join(path_to_cam_dir, "colmap_depth")
        self.path_to_normal_dir = os.path.join(path_to_cam_dir, "colmap_normals")
        self.path_to_image_dir = os.path.join(path_to_cam_dir, "images")
        assert os.path.exists(self.path_to_consistency_dir), f"{self.path_to_consistency_dir} directory does not exist!"
        assert os.path.exists(self.path_to_depth_dir), f"{self.path_to_depth_dir} directory does not exist!"
        assert os.path.exists(self.path_to_normal_dir), f"{self.path_to_normal_dir} directory does not exist!"
        assert os.path.exists(self.path_to_image_dir), f"{self.path_to_image_dir} directory does not exist!"
        if self.has_gt_landmarks:
            self.path_to_landmarks_dir = os.path.join(path_to_cam_dir, "landmarks")
            assert os.path.exists(self.path_to_landmarks_dir), f"{self.path_to_landmarks_dir} directory does not exist!"
        # compute consistent frame ids and store them in a list (self.frame_ids)
        self._compute_consistent_frame_ids()

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx: int) -> dict:
        features = {}
        features["consistency"] = np.load(os.path.join(self.path_to_consistency_dir, f"{self.frame_ids[idx]}.npy"))
        features["depth"] = np.load(os.path.join(self.path_to_depth_dir, f"{self.frame_ids[idx]}.npy"))
        features["normals"] = np.load(os.path.join(self.path_to_normal_dir, f"{self.frame_ids[idx]}.npy"))
        features["image"] = cv2.imread(os.path.join(self.path_to_image_dir, f"{self.frame_ids[idx]}.png"))
        if self.has_gt_landmarks:
            features["landmarks"] = np.load(os.path.join(self.path_to_landmarks_dir, f"{self.frame_ids[idx]}.npy"))
        return features