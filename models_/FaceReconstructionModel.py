import torch
import torch.nn as nn
import numpy as np
import argparse
import nvdiffrast.torch as dr
import pickle
import torch.nn.functional as F
import pyvista as pv

from tqdm import tqdm
from typing import Tuple
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from torch.optim.lr_scheduler import ExponentialLR

from flame.FLAME import FLAME, FLAMETex
from utils_.transform import intrinsics_to_projection
from utils_.utils import resize_long_side, scale_intrinsic_matrix
from utils_.loss import *
from pytorch3d.ops import sample_points_from_meshes
from kornia.filters.sobel import spatial_gradient


class FaceReconModel(nn.Module):
    def __init__(
        self,
        face_model_config: argparse.Namespace,
        orig_img_shape: Tuple[int, int],
        device: str = "cuda",
    ):
        self.counter = 0

        super(FaceReconModel, self).__init__()
        self.device = device

        # Load FLAME model
        self.face_model = FLAME(face_model_config)
        self.texture_model = FLAMETex(face_model_config)

        # Load FLAME faces
        self.faces = torch.from_numpy(self.face_model.faces.astype(np.int32)).int().to(self.device).contiguous()

        # Define which cameras are used for optimization
        self.depth_camera_ids = torch.tensor(face_model_config.depth_camera_ids)
        self.rgb_camera_ids = torch.tensor(face_model_config.rgb_camera_ids)
        self.landmark_camera_id = torch.tensor(face_model_config.landmark_camera_id)
        self.scan_2_mesh_camera_ids = torch.tensor(face_model_config.scan_2_mesh_camera_ids)
        
        # 3DMM parameters
        self.shape_coeffs = nn.Parameter(torch.zeros(1, face_model_config.shape_params).float())
        self.exp_coeffs = nn.Parameter(torch.zeros(1, face_model_config.expression_params).float())
        self.pose_coeffs = nn.Parameter(torch.zeros(1, face_model_config.pose_params).float())
        self.neck_pose_coeffs = nn.Parameter(torch.zeros(1, face_model_config.neck_pose_params).float())
        self.eye_pose_coeffs = nn.Parameter(torch.zeros(1, face_model_config.eye_pose_params).float())
        self.tex_coeffs = nn.Parameter(torch.zeros(1, face_model_config.tex_params).float())
        self.transl_coeffs = nn.Parameter(torch.zeros(1, 3).float().contiguous())
        self.albedo = nn.Parameter(torch.zeros(1, 256, 256, 3).float())

        # Render-related settings
        self.glctx = dr.RasterizeCudaContext(device=device)
        _, faces, aux = load_obj(face_model_config.head_template_mesh_path, load_textures=False, device=self.device)
        self.uvcoords = aux.verts_uvs[None, ...]
        self.uvfaces = faces.textures_idx.int().contiguous()
        self.orig_img_shape = orig_img_shape
        self.light_coeffs = nn.Parameter(torch.zeros(1, 9, 3).float())  # 9 SH coefficients
        pi = np.pi
        self.constant_factors = torch.tensor(
            [1 / np.sqrt(4 * pi), ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
             ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), (pi / 4) * 3 * (np.sqrt(5 / (12 * pi))),
             (pi / 4) * 3 * (np.sqrt(5 / (12 * pi))), (pi / 4) * 3 * (np.sqrt(5 / (12 * pi))),
             (pi / 4) * (3 / 2) * (np.sqrt(5 / (12 * pi))), (pi / 4) * (1 / 2) * (np.sqrt(5 / (4 * pi)))]).float().to(self.device)

        with open(face_model_config.flame_masks_path, 'rb') as f:
            masks = pickle.load(f, encoding='latin1')

            faces = self.faces.cpu()
            uvfaces = self.uvfaces.cpu()

            face_verts = np.concatenate([masks["face"], masks["left_eyeball"], masks["right_eyeball"], masks["neck"]])
            face_verts_only_face = np.concatenate([masks["face"], masks["left_eyeball"], masks["right_eyeball"]])

            face_faces = []
            face_uvfaces = []
            face_faces_only_face = []
            face_uvfaces_only_face = []

            for i, face, in enumerate(faces):
                for face_vert in face:
                    if int(face_vert) in face_verts:
                        face_faces.append(face)
                        face_uvfaces.append(uvfaces[i])

                    if int(face_vert) in face_verts_only_face:
                        face_faces_only_face.append(face)
                        face_uvfaces_only_face.append(uvfaces[i])

            self.face_faces = torch.stack(face_faces).int().to(self.device).contiguous()
            self.face_uvfaces = torch.stack(face_uvfaces).int().to(self.device).contiguous()
            self.face_faces_only_face = torch.stack(face_faces_only_face).int().to(self.device).contiguous()
            self.face_uvfaces_only_face = torch.stack(face_uvfaces_only_face).int().to(self.device).contiguous()

        # Optimize-related settings
        self.coarse2fine_lrs_first_frame = face_model_config.coarse2fine_lrs_first_frame
        self.coarse2fine_lrs_next_frames = face_model_config.coarse2fine_lrs_next_frames
        self.coarse2fine_opt_steps_first_frame = face_model_config.coarse2fine_opt_steps_first_frame
        self.coarse2fine_opt_steps_next_frames = face_model_config.coarse2fine_opt_steps_next_frames

        # Loss-related settings
        self.landmarks_68_weight = face_model_config.landmarks_68_weight
        self.rgb_weight = face_model_config.rgb_weight
        self.p2point_weight = face_model_config.point2point_weight
        self.p2plane_weight = face_model_config.point2plane_weight
        self.shape_reg_weight = face_model_config.shape_regularization_weight
        self.exp_reg_weight = face_model_config.exp_regularization_weight
        self.tex_reg_weight = face_model_config.tex_regularization_weight
        self.chamfer_weight = face_model_config.chamfer_weight
        self.use_chamfer = face_model_config.use_chamfer
        self.shape_fitting_frames = face_model_config.shape_fitting_frames

        # Camera setting
        self.z_near = 0.01
        self.z_far = 10
        self.world_to_cam = None
        self.cam_to_ndc = None
        self.intrinsics = None
        self.extrinsics = None
        self.cameras_rot = None

        self.coarse2fine_resolutions = list(
            map(lambda x: resize_long_side(
                orig_shape=orig_img_shape,
                dest_len=x
            ),
                face_model_config.coarse2fine_resolutions)
        )

    ### Camera-related functions ###
    def set_transformation_matrices_for_optimization(
        self,
        extrinsic_matrices: torch.Tensor,
        intrinsic_matrices: torch.Tensor,
    ) -> None:
        # Get world to cam matrices
        self.world_to_cam = []
        self.extrinsics = []
        self.cameras_rot = []

        inverse = torch.Tensor(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, -1, 0],
             [0, 0, 0, 1]]
        )
        for extrinsic_matrix in extrinsic_matrices:
            self.world_to_cam.append((inverse @ torch.inverse(extrinsic_matrix)).t())
            self.extrinsics.append(extrinsic_matrix.t())
            self.cameras_rot.append(torch.inverse(extrinsic_matrix[:3, :3]))

        self.world_to_cam = torch.stack(self.world_to_cam).float().to(self.device)
        self.extrinsics = torch.stack(self.extrinsics).float().to(self.device)
        self.cameras_rot = torch.stack(self.cameras_rot).float().to(self.device)

        # Get cam to ndc matrices
        self.cam_to_ndc = []
        for resolution in self.coarse2fine_resolutions:
            cam_to_ndc = []
            for intrinsic_matrix in intrinsic_matrices:
                cam_to_ndc.append(intrinsics_to_projection(intrinsic_matrix, resolution).t())
            cam_to_ndc = torch.stack(cam_to_ndc).to(self.device)
            self.cam_to_ndc.append(cam_to_ndc)

        # Get intrinsic matrices
        self.intrinsics = []
        for resolution in self.coarse2fine_resolutions:
            intrinsics = []
            for intrinsic_matrix in intrinsic_matrices:
                intrinsics.append(torch.inverse(scale_intrinsic_matrix(self.orig_img_shape,
                                                                       intrinsic_matrix,
                                                                       resolution)).t())
            intrinsics = torch.stack(intrinsics).to(self.device)
            self.intrinsics.append(intrinsics)

    def set_transformation_matrices_for_fused_point_cloud(
        self,
        extrinsic_matrices: torch.Tensor,
    ) -> None:
        self.extrinsics_fused_point_cloud = []

        for extrinsic_matrix in extrinsic_matrices:
            self.extrinsics_fused_point_cloud.append(extrinsic_matrix.t())

        self.extrinsics_fused_point_cloud = torch.stack(self.extrinsics_fused_point_cloud).float().to(self.device)
    
    def set_initial_pose(
        self,
        first_frame_features: dict
    ):
        camera_ids = first_frame_features["camera_id"]

        rgb_camera_mask = np.isin(camera_ids, self.rgb_camera_ids)
        landmark_camera_mask = np.isin(camera_ids, self.landmark_camera_id)
        rgb_in_landmark_mask = np.isin(camera_ids[rgb_camera_mask], self.landmark_camera_id)

        input_landmarks = first_frame_features["predicted_landmark_3d"][landmark_camera_mask]
        input_landmarks = input_landmarks[:, 17:].float().to(self.device)

        not_nan_indices = ~(torch.isnan(input_landmarks).any(dim=-1))
        input_landmarks = input_landmarks[not_nan_indices]

        for _, lr, opt_steps, _ in zip(self.coarse2fine_resolutions, self.coarse2fine_lrs_first_frame,
                                       self.coarse2fine_opt_steps_first_frame, self.cam_to_ndc):
            # Create optimizer
            optimizer = torch.optim.Adam(
                [{'params': [self.pose_coeffs, self.transl_coeffs]}],
                lr=lr,
            )
            scheduler = ExponentialLR(optimizer, gamma=0.999)

            for _ in tqdm(range(opt_steps)):
                _, landmarks_68_model, landmarks_mp_model = self.face_model(
                    shape_params=self.shape_coeffs,
                    expression_params=self.exp_coeffs,
                    pose_params=self.pose_coeffs,
                    transl=self.transl_coeffs,
                    cameras_rot=self.cameras_rot[rgb_in_landmark_mask],
                )
                landmarks_68_model = landmarks_68_model[:, 17:][not_nan_indices][None, ...]
                landmarks_68_loss = landmark_distance(input_landmarks, landmarks_68_model, torch.ones(landmarks_68_model.shape[0], landmarks_68_model.shape[1]).to(self.device))

                loss = self.landmarks_68_weight * landmarks_68_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

    ### Space transformation functions ###
    def _backproject_to_world_space(
        self,
        xys_cam: torch.Tensor,
        depths: torch.Tensor,
        extrinsics: torch.Tensor,
    ):
        # Convert to cam space
        xyzs_cam = depths * xys_cam

        # Convert to world space
        xyzs_cam = torch.concatenate([xyzs_cam, torch.ones(xyzs_cam.shape[0], xyzs_cam.shape[1], 1).to(self.device)],
                                     dim=-1)
        xyzs_world = torch.matmul(xyzs_cam, extrinsics)
        return xyzs_world[:, :, :3]

    def _backproject_to_world_space_normal(
        self,
        normals_cam: torch.Tensor,
    ):
        # Undo the rotation of the normals
        normals_world = torch.matmul(normals_cam, self.extrinsics[:, :3, :3])
        return normals_world

    def _project_to_cam_space(
        self,
        vertices_world: torch.Tensor,
        landmarks_68_world: torch.Tensor,
        landmarks_mp_world: torch.Tensor,
        world_to_cam: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vertices = torch.cat((vertices_world, torch.ones(vertices_world.shape[0], vertices_world.shape[1], 1).to(self.device)), dim=-1)
        landmarks_68 = torch.cat((landmarks_68_world, torch.ones(landmarks_68_world.shape[0], landmarks_68_world.shape[1], 1).to(self.device)),
                                  dim=-1)
        landmarks_mp = torch.cat((landmarks_mp_world, torch.ones(landmarks_mp_world.shape[0], landmarks_mp_world.shape[1], 1).to(self.device)),
                                  dim=-1)
        vertices_cam = torch.matmul(vertices, world_to_cam)
        landmarks_68_cam = torch.matmul(landmarks_68, world_to_cam)
        landmarks_mp_cam = torch.matmul(landmarks_mp, world_to_cam)
        return vertices_cam, landmarks_68_cam, landmarks_mp_cam

    def _project_to_ndc_space(
        self,
        vertices_cam: torch.Tensor,
        cam_to_ndc: torch.Tensor
    ) -> torch.Tensor:
        vertices_ndc = torch.matmul(vertices_cam, cam_to_ndc).contiguous()
        return vertices_ndc

    def _project_to_image_space(
        self,
        pixels_ndc: torch.Tensor,
        resolution: Tuple[int, int]
    ) -> torch.Tensor:
        pixels_screen = pixels_ndc[:, :, :2] / pixels_ndc[:, :, 2][..., None]
        pixels_screen[:, :, 0] = (pixels_screen[:, :, 0] + 1) * resolution[1] * 0.5
        pixels_screen[:, :, 1] = (1 - pixels_screen[:, :, 1]) * resolution[0] * 0.5
        return pixels_screen

    # The below function is taken from https://github.com/HavenFeng/photometric_optimization/blob/master/renderer.py#L207
    def _add_shlight(
        self,
        normal_images: torch.Tensor,
    ) -> torch.Tensor:
        N = normal_images
        sh = torch.stack([
            N[:, 0] * 0. + 1., N[:, 0], N[:, 1],
            N[:, 2], N[:, 0] * N[:, 1], N[:, 0] * N[:, 2],
            N[:, 1] * N[:, 2], N[:, 0] ** 2 - N[:, 1] ** 2, 3 * (N[:, 2] ** 2) - 1], 1).to(self.device)  # [bz, 9, h, w]
        sh = sh * self.constant_factors[None, :, None, None]
        shading = torch.sum(self.light_coeffs[:, :, :, None, None] * sh[:, :, None, :, :], 1)  # [bz, 9, 3, h, w]
        return shading

    def _compute_normals_from_depth(
        self,
        xyz_images: torch.Tensor,
    ) -> torch.Tensor:
        gradients = spatial_gradient(xyz_images)  # Bx3x2xHxW

        # compute normals
        a, b = gradients[:, :, 0], gradients[:, :, 1]  # Bx3xHxW

        normals = torch.cross(a, b, dim=1)  # Bx3xHxW
        return F.normalize(normals, dim=1, p=2)

    def _render(
        self,
        verts_world: torch.Tensor,
        verts_cam: torch.Tensor,
        verts_ndc: torch.Tensor,
        albedos: torch.Tensor,
        resolution: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        rast_out, _ = dr.rasterize(self.glctx, verts_ndc, self.face_faces, resolution=resolution)
        texc, _ = dr.interpolate(self.uvcoords, rast_out, self.face_uvfaces)
        color = dr.texture(albedos, 1 - texc, filter_mode='linear')
        color = color * torch.clamp(rast_out[..., -1:], 0, 1)
        color = color.permute(0, 3, 1, 2)
        mesh = Meshes(verts=verts_world, faces=self.faces[None, ...])
        normal = mesh.verts_normals_packed()
        normal, _ = dr.interpolate(normal[None, ...].contiguous(), rast_out, self.face_faces)
        normal = normal.permute(0, 3, 1, 2)

        shading = self._add_shlight(normal)
        color = color * shading
        color = color.permute(0, 2, 3, 1)

        depth, _ = dr.interpolate(verts_cam[..., 2:3].contiguous(), rast_out, self.face_faces)

        # Get depth and color masks
        depth_mask = (rast_out[..., 3] > 0).float().unsqueeze(1) # the rendered depth mask includes the neck region
        rast_out, _ = dr.rasterize(self.glctx, verts_ndc, self.face_faces_only_face, resolution=resolution)
        color_mask = (rast_out[..., 3] > 0).float().unsqueeze(1)

        return color, depth, depth_mask, color_mask

    def _get_input_rgb_data(
        self,
        resolution: Tuple[int, int],
        frame_features: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb_masks = torch.isin(frame_features["camera_id"], self.rgb_camera_ids)

        input_rgb = frame_features["image"][rgb_masks]
        input_rgb_pixel_mask = frame_features["pixel_mask"][rgb_masks]

        input_rgb = input_rgb.float().to(self.device)
        input_rgb_pixel_mask = input_rgb_pixel_mask.float().to(self.device)

        input_rgb_combined = torch.concatenate([input_rgb, input_rgb_pixel_mask], dim=1)
        input_rgb_combined_resized = F.interpolate(input_rgb_combined, size=resolution).to(self.device)

        del input_rgb, input_rgb_pixel_mask, input_rgb_combined

        input_rgb_combined_resized = input_rgb_combined_resized.permute(0, 2, 3, 1)

        input_rgb = input_rgb_combined_resized[..., :3].reshape(input_rgb_combined_resized.shape[0], -1, 3)
        input_rgb_pixel_mask = input_rgb_combined_resized[..., -1].reshape(input_rgb_combined_resized.shape[0], -1)

        return input_rgb, input_rgb_pixel_mask

    def _get_input_depth_data(
        self,
        resolution: Tuple[int, int],
        intrinsic: torch.Tensor,
        frame_features: dict,
        rgb_in_depth_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # Get input depth data (pixels backprojected to world space and it's normals) and depth pixel mask
        depth_masks = torch.isin(frame_features["camera_id"], self.depth_camera_ids)

        input_depth = frame_features["depth"][depth_masks]
        input_depth_pixel_mask = frame_features["pixel_mask"][depth_masks]

        input_depth = input_depth.float().to(self.device)
        input_depth_pixel_mask = input_depth_pixel_mask.float().to(self.device)

        input_depth_combined = torch.concatenate([input_depth, input_depth_pixel_mask], dim=1)
        input_depth_combined_resized = F.interpolate(input_depth_combined, size=resolution).to(self.device)

        del input_depth, input_depth_pixel_mask, input_depth_combined
        
        input_depth_combined_resized = input_depth_combined_resized.permute(0, 2, 3, 1)
        input_depth = input_depth_combined_resized[..., :1].reshape(input_depth_combined_resized.shape[0], -1, 1)
        input_depth_pixel_mask = input_depth_combined_resized[..., -1].reshape(input_depth_combined_resized.shape[0], -1)

        # Get depth map with x, y coordinates
        x_indices, y_indices = np.meshgrid(np.arange(resolution[1]), np.arange(resolution[0]))
        x_indices = torch.from_numpy(x_indices.flatten()).to(self.device)[None, ...]
        x_indices = torch.repeat_interleave(x_indices, input_depth_combined_resized.shape[0], 0)[..., None]
        y_indices = torch.from_numpy(y_indices.flatten()).to(self.device)[None, ...]
        y_indices = torch.repeat_interleave(y_indices, input_depth_combined_resized.shape[0], 0)[..., None]
        xy_coordinates = torch.concatenate([x_indices, y_indices], dim=-1)

        # Get 3D coordinates of input pixels in world space
        xys_homo = torch.concatenate(
            [xy_coordinates, torch.ones(xy_coordinates.shape[0], xy_coordinates.shape[1], 1).to(self.device)],
            dim=-1)

        xys_cam = torch.matmul(xys_homo, intrinsic[rgb_in_depth_mask])

        pixels_world_input = self._backproject_to_world_space(xys_cam, input_depth, self.extrinsics[rgb_in_depth_mask])
        normals_world_input = self._compute_normals_from_depth(
            pixels_world_input.reshape(-1, resolution[0], resolution[1], 3).permute(0, 3, 1, 2))
        normals_world_input = normals_world_input.permute(0, 2, 3, 1).reshape(normals_world_input.shape[0], -1, 3)

        return pixels_world_input, normals_world_input, input_depth, input_depth_pixel_mask, xys_cam

    def _get_input_fused_point_cloud(
        self,
        resolution: Tuple[int, int],
        intrinsic: torch.Tensor,
        frame_features: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get input depth data (pixels backprojected to world space and it's normals) and depth pixel mask
        depth_masks = torch.isin(frame_features["camera_id"], self.scan_2_mesh_camera_ids)

        input_depth = frame_features["depth"][depth_masks]
        input_depth_pixel_mask = frame_features["pixel_mask"][depth_masks]

        input_depth = input_depth.float().to(self.device)
        input_depth_pixel_mask = input_depth_pixel_mask.float().to(self.device)

        input_depth_combined = torch.concatenate([input_depth, input_depth_pixel_mask], dim=1)
        input_depth_combined_resized = F.interpolate(input_depth_combined, size=resolution).to(self.device)

        del input_depth, input_depth_pixel_mask, input_depth_combined

        input_depth_combined_resized = input_depth_combined_resized.permute(0, 2, 3, 1)
        input_depth = input_depth_combined_resized[..., :1].reshape(input_depth_combined_resized.shape[0], -1, 1)
        input_depth_pixel_mask = input_depth_combined_resized[..., -1].reshape(input_depth_combined_resized.shape[0], -1)

        # Get depth map with x, y coordinates
        x_indices, y_indices = np.meshgrid(np.arange(resolution[1]), np.arange(resolution[0]))
        x_indices = torch.from_numpy(x_indices.flatten()).to(self.device)[None, ...]
        x_indices = torch.repeat_interleave(x_indices, input_depth_combined_resized.shape[0], 0)[..., None]
        y_indices = torch.from_numpy(y_indices.flatten()).to(self.device)[None, ...]
        y_indices = torch.repeat_interleave(y_indices, input_depth_combined_resized.shape[0], 0)[..., None]
        xy_coordinates = torch.concatenate([x_indices, y_indices], dim=-1)

        # Get 3D coordinates of input pixels in world space
        xys_homo = torch.concatenate(
            [xy_coordinates, torch.ones(xy_coordinates.shape[0], xy_coordinates.shape[1], 1).to(self.device)],
            dim=-1)
        xys_cam = torch.matmul(xys_homo, intrinsic)

        pixels_world_input = self._backproject_to_world_space(xys_cam, input_depth, self.extrinsics_fused_point_cloud)
        normals_world_input = self._compute_normals_from_depth(
            pixels_world_input.reshape(-1, resolution[0], resolution[1], 3).permute(0, 3, 1, 2))
        normals_world_input = normals_world_input.permute(0, 2, 3, 1).reshape(normals_world_input.shape[0], -1, 3)

        return pixels_world_input, normals_world_input, input_depth_pixel_mask

    def _get_input_landmarks_data(
        self,
        resolution: Tuple[int, int],
        frame_features: dict,
    ) -> torch.Tensor:
        landmark_masks = torch.isin(frame_features["camera_id"], self.landmark_camera_id)
        input_2d_landmarks = frame_features["predicted_landmark_2d"][landmark_masks].float().to(self.device)
        input_2d_landmarks[:, :, 0] *= resolution[1] / self.orig_img_shape[1]
        input_2d_landmarks[:, :, 1] *= resolution[0] / self.orig_img_shape[0]

        return input_2d_landmarks

    def optimize(
        self,
        frame_features: dict,
        first_frame: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        color, depth, input_rgb, input_depth, landmarks_68_screen, landmarks_mp_screen, rgb_in_landmarks_masks = torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor()

        for resolution, lr, opt_steps, cam2ndc, intrinsic in zip(self.coarse2fine_resolutions,
                                                                 self.coarse2fine_lrs_first_frame if first_frame else self.coarse2fine_lrs_next_frames,
                                                                 self.coarse2fine_opt_steps_first_frame if first_frame else self.coarse2fine_opt_steps_next_frames,
                                                                 self.cam_to_ndc,
                                                                 self.intrinsics):

            # Get masks for input data
            rgb_masks = torch.isin(frame_features["camera_id"], self.rgb_camera_ids)
            rgb_in_depth_masks = torch.isin(frame_features["camera_id"][rgb_masks], self.depth_camera_ids)
            rgb_in_landmarks_masks = torch.isin(frame_features["camera_id"][rgb_masks], self.landmark_camera_id)

            # Get input data for optimization
            if len(self.rgb_camera_ids) > 0:
                input_rgb, input_rgb_pixel_mask = self._get_input_rgb_data(resolution, frame_features)
            if len(self.depth_camera_ids) > 0:
                pixels_world_input, normals_world_input, input_depth, input_depth_pixel_mask, xys_cam = self._get_input_depth_data(resolution, intrinsic, frame_features, rgb_in_depth_masks)
                if self.use_chamfer:
                    pixels_world = []
                    normals_world = []
                    for i in range(pixels_world_input.shape[0]):
                        pixels_world.append(pixels_world_input[i][input_depth_pixel_mask[i] > 0])
                        normals_world.append(normals_world_input[i][input_depth_pixel_mask[i] > 0])
                    
                    pixels_world_input = torch.concatenate(pixels_world)
                    normals_world_input = torch.concatenate(normals_world)

            input_2d_landmarks = self._get_input_landmarks_data(resolution, frame_features)

            # Create optimizer
            params = [self.exp_coeffs, self.pose_coeffs, self.tex_coeffs, self.transl_coeffs, self.light_coeffs,
                      self.albedo, self.neck_pose_coeffs, self.eye_pose_coeffs]

            if self.counter < self.shape_fitting_frames:
                params.append(self.shape_coeffs)

            optimizer = torch.optim.Adam(
                [{'params': params}],
                lr=lr,
            )
            scheduler = ExponentialLR(optimizer, gamma=0.999)

            # Optimize
            for _ in tqdm(range(opt_steps)):
                # Get vertices in world space and compute vertices attributes
                vertices_world, landmarks_68_world, landmarks_mp_world = self.face_model(
                    shape_params=self.shape_coeffs,
                    expression_params=self.exp_coeffs,
                    pose_params=self.pose_coeffs,
                    transl=self.transl_coeffs,
                    neck_pose=self.neck_pose_coeffs,
                    cameras_rot=self.cameras_rot[rgb_in_landmarks_masks],
                    eye_pose=self.eye_pose_coeffs,
                )

                # Get vertices in camera space
                vertices_cam, landmarks_68_cam, landmarks_mp_cam = self._project_to_cam_space(vertices_world,
                                                                                              landmarks_68_world,
                                                                                              landmarks_mp_world,
                                                                                              self.world_to_cam)

                # Get albedos from the texture model
                albedos = (self.texture_model(self.tex_coeffs) / 255.0).permute(0, 2, 3, 1).contiguous()

                # Get landmarks in screen space
                vertices_ndc = self._project_to_ndc_space(vertices_cam, cam2ndc)

                landmarks_68_ndc = self._project_to_ndc_space(landmarks_68_cam, cam2ndc)
                landmarks_mp_ndc = self._project_to_ndc_space(landmarks_mp_cam, cam2ndc)
                landmarks_68_ndc[:, :, 1] = -landmarks_68_ndc[:, :, 1]
                landmarks_mp_ndc[:, :, 1] = -landmarks_mp_ndc[:, :, 1]
                landmarks_68_screen = self._project_to_image_space(landmarks_68_ndc, resolution)[rgb_in_landmarks_masks]
                landmarks_mp_screen = self._project_to_image_space(landmarks_mp_ndc, resolution)[rgb_in_landmarks_masks]

                # Render images from different cameras
                color, depth, depth_pixel_mask, color_pixel_mask = self._render(vertices_world, vertices_cam,
                                                                                vertices_ndc, albedos, resolution)

                # Compute losses
                rgb_loss, depth_loss = torch.tensor(0), torch.tensor(0)

                # Compute rgb loss
                if len(self.rgb_camera_ids) > 0:
                    color = color.reshape(color.shape[0], -1, 3)
                    color_pixel_mask = color_pixel_mask.reshape(color_pixel_mask.shape[0], -1)
                    color_pixel_mask *= input_rgb_pixel_mask
                    num_valid_pixels_color = (color_pixel_mask > 0).sum(-1)
                    rgb_loss = pixel2pixel_distance(input_rgb, color, color_pixel_mask, num_valid_pixels_color)
                    rgb_loss = rgb_loss * self.rgb_weight

                # Compute depth loss
                if len(self.depth_camera_ids > 0):
                    print("Computing depth loss")
                    depth = depth[rgb_in_depth_masks]
                    depth = -depth.reshape(depth.shape[0], -1, 1)

                    pixels_world_rendered = self._backproject_to_world_space(xys_cam, depth, self.extrinsics[rgb_in_depth_masks])
                    normals_world_rendered = self._compute_normals_from_depth(
                        pixels_world_rendered.reshape(-1, resolution[0], resolution[1], 3).permute(0, 3, 1, 2))
                    normals_world_rendered = normals_world_rendered.permute(0, 2, 3, 1).reshape(
                        normals_world_rendered.shape[0], -1, 3)

                    depth_pixel_mask = depth_pixel_mask[rgb_in_depth_masks]
                    depth_pixel_mask = depth_pixel_mask.reshape(depth_pixel_mask.shape[0], -1)
                    depth_pixel_mask *= input_depth_pixel_mask
                    num_valid_pixels_depth = (depth_pixel_mask > 0).sum(-1)

                    if not self.use_chamfer:
                        p2point_loss = point2point_distance(pixels_world_input, pixels_world_rendered, depth_pixel_mask,
                                                            num_valid_pixels_depth, threshold=0.005)
                        p2plane_loss = point2plane_distance(pixels_world_input, normals_world_input,
                                                            pixels_world_rendered,
                                                            normals_world_rendered, depth_pixel_mask,
                                                            num_valid_pixels_depth, threshold=0.005)
                        depth_loss = self.p2point_weight * p2point_loss + self.p2plane_weight * p2plane_loss
                    else:
                        # This function currently does not work as expected
                        # Get the mesh
                        mesh = Meshes(verts=vertices_world, faces=self.face_faces_only_face[None, ...])
                        scan_to_mesh_loss, _ = scan_to_mesh_distance(pixels_world_input, normals_world_input,
                                                                     *sample_points_from_meshes(mesh, num_samples=5000,
                                                                                                return_normals=True),
                                                                     threshold=0.000025)
                        scan_to_mesh_loss = torch.sqrt(scan_to_mesh_loss)
                        depth_loss = self.chamfer_weight * scan_to_mesh_loss

                # Compute landmarks loss
                landmarks_68_mask = torch.ones(landmarks_68_screen.shape[:2]).to(self.device)
                landmarks_68_loss = landmark_distance(input_2d_landmarks, landmarks_68_screen, landmarks_68_mask)
                landmarks_68_loss = landmarks_68_loss * self.landmarks_68_weight

                # Compute total loss
                loss = landmarks_68_loss + \
                       depth_loss + \
                       rgb_loss + \
                       self.shape_reg_weight * self.shape_coeffs.norm(p=2, dim=1) + \
                       self.eye_pose_coeffs.norm(p=2, dim=1) + \
                       self.exp_reg_weight * self.exp_coeffs.norm(p=2, dim=1) + \
                       self.tex_reg_weight * self.tex_coeffs.norm(p=2, dim=1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

        color = color.reshape(color.shape[0], *self.coarse2fine_resolutions[-1], 3)
        if len(self.depth_camera_ids) > 0:
            depth = depth.reshape(depth.shape[0], *self.coarse2fine_resolutions[-1], 1)
        input_rgb = input_rgb.reshape(input_rgb.shape[0], *self.coarse2fine_resolutions[-1], 3)
        if len(self.depth_camera_ids) > 0:
            input_depth = input_depth.reshape(input_depth.shape[0], *self.coarse2fine_resolutions[-1], 1)

        # Compute the mesh to scan loss
        pixels_world_input, normals_world_input, input_depth_pixel_mask = self._get_input_fused_point_cloud(resolution, intrinsic[:len(self.scan_2_mesh_camera_ids)], frame_features)
        pixels_world = []
        normals_world = []
        for i in range(pixels_world_input.shape[0]):
            pixels_world.append(pixels_world_input[i][input_depth_pixel_mask[i] > 0])
            normals_world.append(normals_world_input[i][input_depth_pixel_mask[i] > 0])
        
        pixels_world_input = torch.concatenate(pixels_world)
        normals_world_input = torch.concatenate(normals_world)

        mesh = Meshes(verts=vertices_world, faces=self.face_faces_only_face[None, ...])
        scan_to_mesh_loss, _ = scan_to_mesh_distance(pixels_world_input[None, ...], normals_world_input[None, ...],
                                                     *sample_points_from_meshes(mesh, num_samples=5000,
                                                                                return_normals=True),
                                                     threshold=0.000025)
        scan_to_mesh_loss = torch.sqrt(scan_to_mesh_loss)

        print("Scan to mesh loss: ", scan_to_mesh_loss)
        # increase the counter
        self.counter += 1
        return color, depth, input_rgb, input_depth, landmarks_68_screen, landmarks_mp_screen, rgb_in_landmarks_masks

