import torch
from pathlib import Path
import numpy as np

from torch.utils.data import DataLoader
from torchvision.transforms.transforms import Compose as TransformCompose

from config import get_config

from dataset.CameraFrameDataset import CameraFrameDataset
from dataset.transforms import ToTensor

from models_.FaceReconstructionModel import FaceReconModel
from models_.HairSegmenter import HairSegmenter
from models_.LandmarkDetectorPIPNET import LandmarkDetectorPIPENET

"""
  Generate the vertices and texture data needed to train MVP
"""

if __name__ == '__main__':
    config = get_config(path_to_data_dir="../")

    # data loading
    landmark_detector = LandmarkDetectorPIPENET()
    hair_segmentor = HairSegmenter(config.hair_segmenter_path)

    transforms = TransformCompose([ToTensor()])
    dataset = CameraFrameDataset(config.cam_data_dir, need_backprojection=True, has_gt_landmarks=True,
                                 transform=transforms)
    dataset.precompute_landmarks(landmark_detector, force_precompute=False)
    dataset.precompute_hair_masks(hair_segmentor, force_precompute=False)

    dataloader = DataLoader(dataset, batch_size=dataset.num_cameras(), shuffle=False, num_workers=0)
    # get camera settings
    first_frame_features = next(iter(dataloader))

    rgb_camera_ids = config.rgb_camera_ids
    rgb_camera_mask = np.isin(first_frame_features["camera_id"], rgb_camera_ids)
    extrinsic_matrices_optimization = first_frame_features["extrinsics"][rgb_camera_mask]
    intrinsic_matrices_optimization = first_frame_features["intrinsics"][rgb_camera_mask]

    mesh_2_scan_camera_ids = config.scan_2_mesh_camera_ids
    mesh_2_scan_camera_mask = np.isin(first_frame_features["camera_id"], mesh_2_scan_camera_ids)
    extrinsic_matrices_fused_point_cloud = first_frame_features["extrinsics"][mesh_2_scan_camera_mask]

    # get face reconstruction model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    face_recon_model = FaceReconModel(
        face_model_config=config,
        orig_img_shape=first_frame_features["image"].shape[2:],
        device=device,
    )
    face_recon_model.to(device)

    # set transformation matrices for projection
    face_recon_model.set_transformation_matrices_for_optimization(
        extrinsic_matrices=extrinsic_matrices_optimization,
        intrinsic_matrices=intrinsic_matrices_optimization,
    )

    face_recon_model.set_transformation_matrices_for_fused_point_cloud(
        extrinsic_matrices=extrinsic_matrices_fused_point_cloud,
    )

    # compute the initial alignment
    face_recon_model.set_initial_pose(first_frame_features)


    # save .obj file
    obj_file_path = config.mesh_data_dir + "/flame.obj"
    Path(obj_file_path).parent.mkdir(parents=True, exist_ok=True)
    vertices, faces, uv_coords = face_recon_model.get_flame()

    face_verts_uv_coords = []
    for face in faces:
        for vertex in face:
            face_verts_uv_coords.append(uv_coords[vertex])

    with open(obj_file_path, 'w') as file:
        # Write vertices
        for vertex in vertices:
            file.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')

        # Write UV coordinates
        for uv in face_verts_uv_coords:
            file.write(f'vt {uv[0]} {uv[1]}\n')

        # Write faces
        counter = 1
        for face in faces:
            file.write(f'f {face[0]+1}/{counter} {face[1]+1}/{counter+1} {face[2]+1}/{counter+2}\n')
            counter += 3

    sequence_verts = []
    for frame_num, frame_features in enumerate(dataloader):
        frame_id = frame_features["frame_id"][0]
        landmark_mask = np.isin(frame_features["camera_id"], config.landmark_camera_id)
        _, _, _, _, _, _, _, _, flame_vertices = face_recon_model.optimize(
            frame_features, first_frame=frame_num == 0)
        flame_vertices = flame_vertices.detach().cpu().numpy().squeeze().astype(np.float32)
        bin_file_path = config.mesh_data_dir + f"/{frame_id}.bin"
        flame_vertices.tofile(bin_file_path)
        sequence_verts.append(flame_vertices)

    sequence_verts = np.stack(sequence_verts, axis=0)
    verts_mean = np.mean(sequence_verts, axis=0)
    verts_std = np.std(sequence_verts)
    ver_mean_bin_file_path = config.mesh_data_dir + f"/vert_mean.bin"
    verts_mean.tofile(ver_mean_bin_file_path)
    with open(config.mesh_data_dir + f"/vert_var.txt", 'w') as file:
        file.write(f'{verts_std**2}')