import torch
import torch.nn as nn
import numpy as np
import argparse
import nvdiffrast.torch as dr

from flame.FLAME import FLAME, FLAMETex
from typing import Tuple
from utils.transform import intrinsics_to_projection
from utils.utils import resize_long_side
from utils.transform import rigid_transform_3d, rotation_matrix_to_axis_angle


class FaceReconModel(nn.Module):
    def __init__(
        self,
        face_model_config: argparse.Namespace,
        orig_img_shape: tuple[int, int],
        device: str = "cuda",
    ):
        super(FaceReconModel, self).__init__()
        self.device = device

        self.face_model = FLAME(face_model_config)
        self.texture_model = FLAMETex(face_model_config)

        self.faces = torch.from_numpy(self.face_model.faces.astype(np.int32)).int().to(self.device).contiguous()

        # 3DMM parameters
        self.shape_coeffs = nn.Parameter(torch.zeros(1, face_model_config.shape_params).float())
        self.exp_coeffs = nn.Parameter(torch.zeros(1, face_model_config.expression_params).float())
        self.pose_coeffs = nn.Parameter(torch.zeros(1, face_model_config.pose_params).float())
        self.tex_coeffs = nn.Parameter(torch.zeros(1, face_model_config.tex_params).float())

        # Render-related settings
        self.glctx = dr.RasterizeCudaContext(device=device)

        # Optimize-related settings
        self.coarse2fine_lrs = face_model_config.coarse2fine_lrs
        self.coarse2fine_opt_steps = face_model_config.coarse2fine_opt_steps

        # Loss-related settings
        self.land_weight = face_model_config.landmark_weight
        self.rgb_weight = face_model_config.rgb_weight
        self.p2point_weight = face_model_config.point2point_weight
        self.p2plane_weight = face_model_config.point2plane_weight
        self.shape_reg_weight = face_model_config.shape_regularization_weight
        self.exp_reg_weight = face_model_config.exp_regularization_weight
        self.tex_reg_weight = face_model_config.tex_regularization_weight

        # Camera setting
        self.z_near = 0.01
        self.z_far = 10
        self.world_to_cam = None
        self.cam_to_ndc = None

        self.coarse2fine_resolutions = list(map(lambda x: resize_long_side(
                                                            orig_shape=orig_img_shape,
                                                            dest_len=x
                                            ),
                                           face_model_config.coarse2fine_resolutions))

    def set_transformation_matrices(
        self,
        extrinsic_matrices: torch.Tensor,
        intrinsic_matrices: torch.Tensor,
    ) -> None:
        # Get world to cam matrices
        self.world_to_cam = []

        for extrinsic_matrix in extrinsic_matrices:
            self.world_to_cam.append((torch.inverse(extrinsic_matrix)).t())
        self.world_to_cam = torch.stack(self.world_to_cam).float().to(self.device)

        # Get cam to ndc matrices
        self.cam_to_ndc = []
        for resolution in self.coarse2fine_resolutions:
            cam_to_ndc = []
            for intrinsic_matrix in intrinsic_matrices:
                cam_to_ndc.append(intrinsics_to_projection(intrinsic_matrix, resolution).t())
            cam_to_ndc = torch.stack(cam_to_ndc).to(self.device)
            self.cam_to_ndc.append(cam_to_ndc)

    def set_initial_pose(
        self,
        input_landmarks: torch.Tensor
    ):
        _, landmarks = self.face_model(
            shape_params=self.shape_coeffs,
            expression_params=self.exp_coeffs,
            pose_params=self.pose_coeffs
        )

        input_landmarks = input_landmarks.cpu().numpy()
        landmarks = landmarks[0].detach().cpu().numpy()

        not_nan_indices = ~(np.isnan(input_landmarks).any(axis=1))
        input_landmarks = input_landmarks[not_nan_indices]
        landmarks = landmarks[not_nan_indices]

        r, t = rigid_transform_3d(input_landmarks.T, landmarks.T)
        r = rotation_matrix_to_axis_angle(r)

        self.pose_coeffs = nn.Parameter(torch.from_numpy(np.concatenate([r, t[:, 0]])[None, ...]).float().to(self.exp_coeffs.device))

    def get_landmarks(self) -> torch.Tensor:
        _, landmarks = self.face_model(
            shape_params=self.shape_coeffs,
            expression_params=self.exp_coeffs,
            pose_params=self.pose_coeffs
        )
        return landmarks

    def _project_to_cam_space(
        self,
        world_to_cam: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vertices, landmarks = self.face_model(
            shape_params=self.shape_coeffs,
            expression_params=self.exp_coeffs,
            pose_params=self.pose_coeffs
        )

        vertices = torch.cat((vertices, torch.ones(vertices.shape[0], vertices.shape[1], 1).to(self.device)), dim=-1)
        landmarks = torch.cat((landmarks, torch.ones(landmarks.shape[0], landmarks.shape[1], 1).to(self.device)), dim=-1)

        vertices_cam = torch.matmul(vertices, world_to_cam)
        vertices_cam[..., 0:3] = -vertices_cam[..., 0:3]
        landmarks_cam = torch.matmul(landmarks, world_to_cam)
        landmarks_cam[..., 0:3] = -landmarks_cam[..., 0:3]
        return vertices_cam, landmarks_cam

    def _project_to_ndc_space(
        self,
        vertices_cam: torch.Tensor,
        landmarks_cam: torch.Tensor,
        cam_to_ndc: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vertices_ndc = torch.matmul(vertices_cam, cam_to_ndc).contiguous()
        landmarks_ndc = torch.matmul(landmarks_cam, cam_to_ndc).contiguous()
        return vertices_ndc, landmarks_ndc

    def _render(
        self,
        verts_cam: torch.Tensor,
        verts_ndc: torch.Tensor,
        albedos: torch.Tensor,
        resolution: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rast_out, _ = dr.rasterize(self.glctx, verts_ndc, self.faces, resolution=resolution)
        color, _ = dr.interpolate(albedos, rast_out, self.faces)
        color = color.permute(0, 3, 1, 2)

        depth, _ = dr.interpolate(verts_cam[..., 2:3].contiguous(), rast_out, self.faces)
        depth = depth.permute(0, 3, 1, 2)

        return color, depth

    def optimize(
        self,
        frame_features: dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Get vertices in cam space and compute vertices attributes
        vertices_cam, landmarks_cam = self._project_to_cam_space(self.world_to_cam)
        # albedos = self.texture_model(self.tex_coeffs) / 255.
        albedos = torch.rand(1, vertices_cam.shape[1], 3).to(self.device)

        for resolution, lr, opt_steps, cam2ndc in zip(self.coarse2fine_resolutions, self.coarse2fine_lrs, self.coarse2fine_opt_steps, self.cam_to_ndc):
            vertices_ndc, landmarks_ndc = self._project_to_ndc_space(vertices_cam, landmarks_cam, cam2ndc)
            color, depth = self._render(vertices_cam, vertices_ndc, albedos, resolution)

        return color, depth




