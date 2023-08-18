import fire
import shutil
import json
import numpy as np
import cv2

from tqdm import tqdm
from pathlib import Path

# Sample usecase:
#
# python scripts/process_raw_sequences.py --root_path ~/sequences/sequence_0 --dest_path ~/3DSSL-SS23_3DMMFitting/data/sequence_0
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
) -> None:
    root_folder = Path(root_path)
    dest_folder = Path(dest_path)

    if not dest_folder.exists():
        dest_folder.mkdir(parents=True)

    sequences_folder = root_folder / "timesteps"

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

        dest_alpha_map_folder = dest_folder_camera / "alpha_map"
        if not dest_alpha_map_folder.exists():
            dest_alpha_map_folder.mkdir(parents=True)

    ## Process frames
    print("Processing frames...")
    for frame_folder in tqdm(sequences_folder.iterdir()):
        frame_num = frame_folder.stem.split("_")[1]
        colmap_folder = frame_folder / "colmap"
        colmap_depth_folder = colmap_folder / "depth_maps_geometric" / "16"

        images_folder = frame_folder / "images-2x"
        alpha_map_folder = frame_folder / "alpha_map"

        ### Copy alpha maps
        for image in alpha_map_folder.iterdir():
            camera_id = image.stem.split("_")[1]
            if camera_id in VALID_CAMERA_IDS:
                dest_image_path = dest_folder / camera_id / "alpha_map" / f"{frame_num}.png"
                if not dest_image_path.exists():
                    shutil.copy(image, dest_image_path)

        ### Copy images
        for image in images_folder.iterdir():
            if image.suffix != ".png":
                continue
            camera_id = image.stem.split("_")[1]
            if camera_id in VALID_CAMERA_IDS:
                dest_image_path = dest_folder / camera_id / "images" / f"{frame_num}.png"
                if not dest_image_path.exists():
                    shutil.copy(image, dest_image_path)

        ### Copy colmap depth
        for depth_map in colmap_depth_folder.iterdir():
            camera_id = depth_map.stem.split("_")[1]
            if camera_id in VALID_CAMERA_IDS:
                dest_depth_map_path = dest_folder / camera_id / "colmap_depth" / f"{frame_num}"
                dest_consistency_graph_path = dest_folder / camera_id / "colmap_consistency" / f"{frame_num}"

                if not dest_depth_map_path.exists():
                    depth_map = np.load(str(depth_map))["arr_0"]
                    colmap_consistency = np.zeros_like(depth_map)
                    colmap_consistency[depth_map > 0] = 1
                    np.save(str(dest_depth_map_path), depth_map)
                    np.save(str(dest_consistency_graph_path), colmap_consistency.astype(np.uint8))

    # Load extrinsics and intrinsics
    intrinsics = np.load(str(root_folder / "intrinsics.npz"))
    extrinsics = np.load(str(root_folder / "c2ws.npz"))

    # Compute intrinsics and extrinsics for each camera
    print("Processing intrinsics and extrinsics...")
    for camera_id in tqdm(VALID_CAMERA_IDS):
        dest_camera_folder = dest_folder / camera_id

        dest_intrinsics_path = dest_camera_folder / "intrinsics.npy"
        np.save(str(dest_intrinsics_path), intrinsics[str(camera_id)])

        dest_extrinsics_path = dest_camera_folder / "extrinsics.npy"
        np.save(str(dest_extrinsics_path), extrinsics[str(camera_id)])

if __name__ == '__main__':
    fire.Fire(run)
