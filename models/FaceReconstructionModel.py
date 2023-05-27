import torch
import torch.nn as nn
import numpy as np
import argparse
import nvdiffrast.torch as dr

from enum import Enum
from flame.FLAME import FLAME, FLAMETex
from typing import Union, Tuple


class FittingMode(Enum):
    IMAGE = 0
    VIDEO = 1


class FaceReconModel(nn.Module):
    def __init__(
        self,
        face_model_config: argparse.Namespace,
        resolution: Union[int, tuple[int, int]],
        device: str = "cuda",
        mode: FittingMode = FittingMode.VIDEO,
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
        self.num_opt_steps_ff = face_model_config.num_opt_steps_first_frame
        self.lr_ff = face_model_config.learning_rate_first_frame

        self.num_opt_steps_sf = face_model_config.num_opt_steps_subsequent_frames
        self.lr_sf = face_model_config.learning_rate_subsequent_frames

        # Loss-related settings
        self.land_weight = face_model_config.landmark_weight
        self.rgb_weight = face_model_config.rgb_weight
        self.p2point_weight = face_model_config.point2point_weight
        self.p2plane_weight = face_model_config.point2plane_weight
        self.shape_weight = face_model_config.shape_weight
        self.exp_weight = face_model_config.exp_weight
        self.tex_weight = face_model_config.tex_weight

        # Camera setting
        self.z_near = 0.01
        self.z_far = 10
        self.resolution = resolution

        # Fitting mode
        self.fitting_mode = mode
        self.is_first_frame = True

    def set_transformation_matrices(
        self,
        extrinsic_matrices: torch.Tensor,
        intrinsic_matrices: torch.Tensor,
        aspect_ratio: float
    ) -> None:
        self.world_to_cam = []
        self.cam_to_ndc = []

        for extrinsic_matrix in extrinsic_matrices:
            print(extrinsic_matrix)
            self.world_to_cam.append(torch.inverse(extrinsic_matrix).t())

        for intrinsic_matrix in intrinsic_matrices:

            intrinsic_matrix_transformed = torch.tensor(
                [[intrinsic_matrix[0][0]/112., 0, 0, 0],
                 [0, intrinsic_matrix[1][1]/112., 0, 0],
                 [0, 0, -(self.z_far+self.z_near)/(self.z_far-self.z_near), -2*(self.z_far*self.z_near)/(self.z_far-self.z_near)],
                 [0, 0, -1, 0]]
            )

            self.cam_to_ndc.append(intrinsic_matrix_transformed.t())

        self.world_to_cam = torch.stack(self.world_to_cam).float().to(self.device)
        self.cam_to_ndc = torch.stack(self.cam_to_ndc).to(self.device)

    def _project(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vertices, landmarks = self.face_model(
            shape_params=self.shape_coeffs,
            expression_params=self.exp_coeffs,
            pose_params=self.pose_coeffs
        )

        vertices = torch.cat((vertices, torch.ones(vertices.shape[0], vertices.shape[1], 1).to(self.device)), dim=-1)
        landmarks = torch.cat((landmarks, torch.ones(landmarks.shape[0], landmarks.shape[1], 1).to(self.device)), dim=-1)

        vertices_projected = torch.matmul(torch.matmul(vertices, self.world_to_cam), self.cam_to_ndc).contiguous()
        landmarks_projected = torch.matmul(torch.matmul(landmarks, self.world_to_cam), self.cam_to_ndc).contiguous()
        print(vertices_projected)
        return vertices_projected, landmarks_projected

    def _render(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        verts, landmarks = self._project()

        #albedos = self.texture_model(self.tex_coeffs) / 255.
        albedos = torch.rand(1, verts.shape[1], 3).to(self.device)
        rast_out, _ = dr.rasterize(self.glctx, verts, self.faces, resolution=self.resolution)
        color, _ = dr.interpolate(albedos, rast_out, self.faces)
        color = color.permute(0, 3, 1, 2)

        depth, _ = dr.interpolate(verts[..., 2:3].contiguous(), rast_out, self.faces)
        depth = depth.permute(0, 3, 1, 2)

        return color, depth

    def optimize(
        self,
        frame_features: dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        color, depth = self._render()
        return color, depth




