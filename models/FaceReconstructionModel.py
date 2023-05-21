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
        device: str = "cuda",
        mode: FittingMode = FittingMode.VIDEO
    ):
        super(FaceReconModel, self).__init__()
        self.face_model = FLAME(face_model_config)
        self.texture_model = FLAMETex(face_model_config)

        self.faces = self.face_model.faces.astype(np.int32)

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

        # Fitting mode
        self.fitting_mode = mode
        self.is_first_frame = True

    def _project(
        self,
        trans_matrices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vertices, landmarks = self.face_model(
            shape_params=self.shape_coeffs,
            expression_params=self.exp_coeffs,
            pose_params=self.pose_coeffs
        )

        vertices_projected = torch.matmul(vertices, trans_matrices.t())
        landmarks_projected = torch.matmul(landmarks, trans_matrices.t())
        return vertices_projected, landmarks_projected

    def _render(
        self,
        trans_matrices: torch.Tensor,
        resolution: Union[Tuple[int, int], int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        verts, landmarks = self._project(trans_matrices)

        albedos = self.flametex(self.tex_coeffs) / 255.
        rast_out, _ = dr.rasterize(self.glctx, verts, self.faces, resolution=resolution)
        color, _ = dr.interpolate(albedos[None, ...], rast_out, self.faces)
        color = color.permute(0, 3, 1, 2)

        depth, _ = dr.interpolate(verts[..., 2].contiguous(), rast_out, self.faces)
        depth = depth.permute(0, 3, 1, 2)

        return color, depth

    def optimize(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        trans_matrices: torch.Tensor,
    ):
        pass



