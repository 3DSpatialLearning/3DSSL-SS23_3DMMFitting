import numpy as np
import os
import cv2

from tqdm import tqdm
from typing import List, Tuple

from torch.utils.data import Dataset
from torch import nn
from torchvision.transforms.transforms import Compose as TransformCompose
from scipy.spatial.distance import cdist

from utils.transform import backproject_points, filter_outliers_landmarks
from models.HairSegmenter import HairSegmenter

"""
    This dataset is used to load a all relevant data related to a sequence of frames from cameras.
    Looping over will retrieve from the list of (cam_id, frame_id) pairs with frame being the outer loop.
    For batching, please make sure to use a batch size same as the number of cameras in the dataset otherwise
    different frame_id will be batched together which is undesirable.
    It expects the following directory structure for all camera folders (file naming have to be consistent):
    - {path_to_data}
        - {cam_id}
            - intrinsics.npy
            - extrinsics.npy
            - colmap_consistency
                - {id}.npy
                ...
            - colmap_depth
                - {id}.npy
                ...
            - images
                - {id}.png
                ...
"""


class CameraFrameDataset(Dataset):
    consistency_threshold = 1

    path_to_data: str = None
    has_gt_landmarks: bool = False
    has_predicted_landmarks: bool = False
    transforms: TransformCompose = None

    consistency_subdir: str = "colmap_consistency"
    depth_subdir: str = "colmap_depth"
    image_subdir: str = "images"
    predicted_2d_landmarks_subdir: str = "predicted_2d_landmarks"
    predicted_3d_landmarks_subdir: str = "predicted_3d_landmarks"
    predicted_hair_masks_subdir: str = "predicted_hair_masks"
    intrinsics_filename: str = "intrinsics.npy"
    extrinsics_filename: str = "extrinsics.npy"

    list_of_cam_folders: List[str] = []
    list_of_frame_ids: List[str] = []
    list_of_cam_frame_pairs: List[Tuple[str, str]] = []

    def __init__(
        self,
        path_to_data: str,
        need_backprojection: bool = False,
        has_gt_landmarks: bool = False,
        consistency_threshold: int = 1,
        transform: TransformCompose = None
    ):
        self.path_to_data = path_to_data
        self.has_gt_landmarks = has_gt_landmarks
        self.transform = transform
        self.consistency_threshold = consistency_threshold
        self.need_backprojection = need_backprojection

        assert os.path.exists(path_to_data), f"{path_to_data} directory does not exist!"
        list_of_cam_folders = os.listdir(path_to_data)
        list_of_cam_folders = [folder for folder in list_of_cam_folders if folder != '.gitkeep']
        self.list_of_cam_folders = sorted(list_of_cam_folders)
        cam_folder = self.list_of_cam_folders[0]

        # check if we have predicted landmarks precomputed
        # assumes all cameras have the same structure, therefore we only check the first one
        cam_folder_subdirs = os.listdir(os.path.join(self.path_to_data, cam_folder))
        self.has_predicted_landmarks = self.predicted_3d_landmarks_subdir in cam_folder_subdirs and self.predicted_2d_landmarks_subdir in cam_folder_subdirs
        # check if we have predicted hair masks precomputed
        self.has_predicted_hair_masks = self.predicted_hair_masks_subdir in cam_folder_subdirs

        # get list of (cam, frame) to iterate over
        self.list_of_frame_ids = sorted(os.listdir(os.path.join(self.path_to_data, cam_folder, self.image_subdir)))
        self.list_of_frame_ids = [os.path.splitext(file)[0] for file in self.list_of_frame_ids if ".png" in file]
        for frame_id in self.list_of_frame_ids:
            for cam_folder in self.list_of_cam_folders:
                self.list_of_cam_frame_pairs.append((cam_folder, frame_id))

    def __len__(self):
        return len(self.list_of_cam_frame_pairs)

    def __getitem__(
        self,
        idx: int
    ) -> dict:
        cam_folder, frame_id = self.list_of_cam_frame_pairs[idx]
        path = os.path.join(self.path_to_data, cam_folder)

        features = {
            "consistency": np.load(os.path.join(path, self.consistency_subdir, f"{frame_id}.npy")),
            "depth": np.load(os.path.join(path, self.depth_subdir, f"{frame_id}.npy")),
            "image": cv2.cvtColor(cv2.imread(os.path.join(path, self.image_subdir, f"{frame_id}.png")), cv2.COLOR_BGR2RGB),
            "intrinsics": np.load(os.path.join(path, self.intrinsics_filename)),
            "extrinsics": np.load(os.path.join(path, self.extrinsics_filename)),
            "camera_id": int(cam_folder),
            "frame_id": frame_id
        }

        if self.has_predicted_landmarks:
            features["predicted_landmark_2d"] = np.load(
                os.path.join(path, self.predicted_2d_landmarks_subdir, f"{frame_id}.npy"))
            features["predicted_landmark_3d"] = np.load(
                os.path.join(path, self.predicted_3d_landmarks_subdir, f"{frame_id}.npy"))
            features["predicted_landmark_3d"] = filter_outliers_landmarks(features["predicted_landmark_3d"],
                                                                          0.03).astype(np.float32)
        # filter out low consistency points
        features["pixel_mask"] = np.where(features["consistency"] < self.consistency_threshold, 0, 1)[..., None]

        # filter out hair points
        hair_mask = np.load(os.path.join(path, self.predicted_hair_masks_subdir, f"{frame_id}.npy"))
        features["pixel_mask"] = np.bitwise_and(features["pixel_mask"], ~hair_mask[..., None])

        if self.transform is not None:
            features = self.transform(features)
        return features

    """
        Precompute landmarks for all frames in all cameras in the dataset using the provided landmark detector if they are not already precomputed
        and saved in disk.
        :param landmark_detector: landmark detector
        :param force_precompute: if True, precompute landmarks even if they are already existing
    """

    def precompute_landmarks(
        self,
        landmark_detector: nn.Module,
        closest_neighbour_distance_threshold: float = 0.05,
        force_precompute: bool = False
    ):
        if not self.has_predicted_landmarks or force_precompute:
            print("Precomputing landmarks...")
            for (cam, frame) in (pbar := tqdm(self.list_of_cam_frame_pairs)):
                pbar.set_description(f"Computing landmarks for camera {cam} and frame {frame}")
                path_to_cam_folder = os.path.join(self.path_to_data, cam)

                image = cv2.imread(os.path.join(path_to_cam_folder, self.image_subdir, f"{frame}.png"))

                predicted_landmarks_2d_dir = os.path.join(path_to_cam_folder, self.predicted_2d_landmarks_subdir)
                predicted_landmarks_3d_dir = os.path.join(path_to_cam_folder, self.predicted_3d_landmarks_subdir)
                if not os.path.exists(predicted_landmarks_2d_dir):
                    os.makedirs(predicted_landmarks_2d_dir)
                if not os.path.exists(predicted_landmarks_3d_dir):
                    os.makedirs(predicted_landmarks_3d_dir)

                # predict landmarks pixel coordinates
                landmark_2d = landmark_detector(image)

                if landmark_2d is not None:
                    # backproject to 3d landmarks
                    depth = np.load(os.path.join(path_to_cam_folder, self.depth_subdir, f"{frame}.npy"))
                    consistency = np.load(os.path.join(path_to_cam_folder, self.consistency_subdir, f"{frame}.npy"))
                    depth[consistency < self.consistency_threshold] = 0
                    
                    camera_intrinsics = np.load(os.path.join(path_to_cam_folder, self.intrinsics_filename))
                    camera_extrinsics = np.load(os.path.join(path_to_cam_folder, self.extrinsics_filename))
                    landmark_3d = backproject_points(landmark_2d, depth, camera_intrinsics, camera_extrinsics)
                    
                    # clear low consistency landmarks
                    landmark_3d[landmark_3d[:, 2] == 0] = [np.nan, np.nan, np.nan]

                    # remove outliers
                    dist_mtrx = np.nan_to_num(cdist(landmark_3d, landmark_3d), nan=np.inf)  # generate distance matrix where nan values (at least one element was nan during the dist calculation) set to inf
                    np.fill_diagonal(dist_mtrx, np.nan)  # set diagonal to nan to not count the distance of an element with itself
                    dist_mtrx[dist_mtrx == 0] = np.nan  # do not consider duplicate point distances
                    mins = np.nanmin(dist_mtrx, axis=1)  # get the closest neighbor distances
                    condition = lambda p, mn : (np.isnan(p).any() or (mn != np.inf and mn > closest_neighbour_distance_threshold))
                    landmark_3d = [[np.nan, np.nan, np.nan] if condition(p, mn) else p for p, mn in zip(landmark_3d, mins)]
                else:
                    landmark_3d = np.zeros((68, 3))
                    landmark_3d[:] = np.nan
                    landmark_2d = np.zeros((68, 2))
                    landmark_2d[:] = np.nan
                np.save(os.path.join(predicted_landmarks_2d_dir, f"{frame}.npy"), landmark_2d)
                np.save(os.path.join(predicted_landmarks_3d_dir, f"{frame}.npy"), landmark_3d)
            self.has_predicted_landmarks = True

    """
        Precompute hair segmentation mask for all frames in all cameras in the dataset using the provided hair segmentor if they are not already precomputed
        and saved in disk.
        :param hair_segmentor: hair segmentor
        :param force_precompute: if True, precompute segmentation map even if they are already existing
    """

    def precompute_hair_masks(
        self,
        hair_segmentor: HairSegmenter,
        force_precompute: bool = False
    ):
        if not self.has_predicted_hair_masks or force_precompute:
            for (cam, frame) in (pbar := tqdm(self.list_of_cam_frame_pairs)):
                pbar.set_description(f"Computing hair mask for camera {cam} and frame {frame}")
                path_to_cam_folder = os.path.join(self.path_to_data, cam)

                image = cv2.imread(os.path.join(path_to_cam_folder, self.image_subdir, f"{frame}.png"))

                predicted_hair_masks_dir = os.path.join(path_to_cam_folder, self.predicted_hair_masks_subdir)
                if not os.path.exists(predicted_hair_masks_dir):
                    os.makedirs(predicted_hair_masks_dir)

                # predict hair mask
                hair_mask = hair_segmentor.segment(image, False)
                np.save(os.path.join(predicted_hair_masks_dir, f"{frame}.npy"), hair_mask)

    def num_cameras(self):
        return len(self.list_of_cam_folders)
