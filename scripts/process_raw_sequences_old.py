import fire
import shutil
import json
import numpy as np
import cv2

from tqdm import tqdm
from pathlib import Path

# Sample usecase:
#
# python scripts/process_raw_sequences.py --root_path ~/Documents/TUM/Projects/3DSSL-SS23_3DMMFitting/data/T1-30-24fps/030/ \
# --dest_path ~/Documents/TUM/Projects/3DSSL-SS23_3DMMFitting/data/subject_0 --exp_name EXP-2-eyes
#
# The exp_name is the name of the folder in the sequences folder
VALID_CAMERA_IDS = ['222200036', '222200037', '222200038', '222200039', '222200041', '222200042',
                    '222200044', '222200045', '222200046', '222200047', '222200048', '222200049']


def axis_angle_to_rotation_matrix(axis_angle: np.ndarray) -> np.ndarray:
    assert axis_angle.shape == (3,), "Axis angle must be 3 dimensional"
    rotation, _ = cv2.Rodrigues(axis_angle)
    return rotation.squeeze()


def run(
        root_path: str,
        dest_path: str,
        exp_name: str
) -> None:
    root_folder = Path(root_path)
    dest_folder = Path(dest_path)

    if not dest_folder.exists():
        dest_folder.mkdir(parents=True)

    calibration_folder = root_folder / "calibration"
    sequences_folder = root_folder / "sequences" / exp_name

    # Process sequences

    ## Create folder structure
    for camera_id in VALID_CAMERA_IDS:
        dest_folder_camera = dest_folder / camera_id
        if not dest_folder_camera.exists():
            dest_folder_camera.mkdir(parents=True)

        dest_colmap_consistency_folder = dest_folder_camera / "colmap_consistency"
        if not dest_colmap_consistency_folder.exists():
            dest_colmap_consistency_folder.mkdir(parents=True)

        dest_colmap_depth_folder = dest_folder_camera / "colmap_depth"
        if not dest_colmap_depth_folder.exists():
            dest_colmap_depth_folder.mkdir(parents=True)

        dest_images_folder = dest_folder_camera / "images"
        if not dest_images_folder.exists():
            dest_images_folder.mkdir(parents=True)

        dest_images_folder = dest_folder_camera / "bg"
        if not dest_images_folder.exists():
            dest_images_folder.mkdir(parents=True)

    ## Process frames
    print("Processing frames...")
    for frame_folder in tqdm(sequences_folder.iterdir()):
        break
        frame_num = frame_folder.stem.split("_")[1]
        colmap_folder = frame_folder / "colmap-73fps"
        colmap_consistency_folder = colmap_folder / "consistency_graphs" / "12"
        colmap_depth_folder = colmap_folder / "depth_maps_geometric" / "12"

        images_folder = frame_folder / "images-73fps"
        bg_folder = frame_folder / "alpha_map-73fps"

        ### Copy background mask
        for image in bg_folder.iterdir():
            camera_id = image.stem.split("_")[1]
            if camera_id in VALID_CAMERA_IDS:
                dest_image_path = dest_folder / camera_id / "bg" / f"{frame_num}.png"
                if not dest_image_path.exists():
                    shutil.copy(image, dest_image_path)

        ### Copy images
        for image in images_folder.iterdir():
            camera_id = image.stem.split("_")[1]
            if camera_id in VALID_CAMERA_IDS:
                dest_image_path = dest_folder / camera_id / "images" / f"{frame_num}.png"
                if not dest_image_path.exists():
                    shutil.copy(image, dest_image_path)

        ### Copy colmap consistency
        for consistency_graph in colmap_consistency_folder.iterdir():
            camera_id = consistency_graph.stem.split("_")[1]
            if camera_id in VALID_CAMERA_IDS:
                dest_consistency_graph_path = dest_folder / camera_id / "colmap_consistency" / f"{frame_num}"
                if not dest_consistency_graph_path.exists():
                    colmap_consistency = np.load(str(consistency_graph))["consistency_graph"]
                    np.save(str(dest_consistency_graph_path), colmap_consistency)

        ### Copy colmap depth
        for depth_map in colmap_depth_folder.iterdir():
            camera_id = depth_map.stem.split("_")[1]
            if camera_id in VALID_CAMERA_IDS:
                dest_depth_map_path = dest_folder / camera_id / "colmap_depth" / f"{frame_num}"
                if not dest_depth_map_path.exists():
                    depth_map = np.load(str(depth_map))["depth_map"]
                    np.save(str(dest_depth_map_path), depth_map)

    # Process calibration
    config_json = calibration_folder / "config.json"
    calibration_result_json = calibration_folder / "calibration_result.json"

    with open(str(config_json), "r") as f:
        config = json.load(f)

    with open(str(calibration_result_json), "r") as f:
        calibration_result = json.load(f)

    # Camera id to r
    camera_id_to_r = {k: np.array(v) for k, v in zip(config["serials"], calibration_result["params_result"]["rs"])}
    # Camera id to t
    camera_id_to_t = {k: np.array(v) for k, v in zip(config["serials"], calibration_result["params_result"]["ts"])}

    intrinsic_params = calibration_result["params_result"]["intrinsics"][0]
    intrinsic_matrix = np.array(
        [[intrinsic_params["fx"], 0, intrinsic_params["cx"]],
         [0, intrinsic_params["fy"], intrinsic_params["cy"]],
         [0, 0, 1]]
    ).astype(np.float32)

    # Compute intrinsics and extrinsics for each camera
    print("Processing intrinsics and extrinsics...")
    for camera_id in tqdm(VALID_CAMERA_IDS):
        dest_camera_folder = dest_folder / camera_id

        dest_intrinsics_path = dest_camera_folder / "intrinsics.npy"
        np.save(str(dest_intrinsics_path), intrinsic_matrix)

        dest_extrinsics_path = dest_camera_folder / "extrinsics.npy"
        r = camera_id_to_r[camera_id]
        t = camera_id_to_t[camera_id]
        extrinsics_matrix = np.eye(4)
        extrinsics_matrix[:3, :3] = axis_angle_to_rotation_matrix(r)
        extrinsics_matrix[:3, 3] = t
        extrinsics_matrix = extrinsics_matrix.astype(np.float32)
        extrinsics_matrix = np.linalg.inv(extrinsics_matrix)
        np.save(str(dest_extrinsics_path), extrinsics_matrix)

if __name__ == '__main__':
    fire.Fire(run)
