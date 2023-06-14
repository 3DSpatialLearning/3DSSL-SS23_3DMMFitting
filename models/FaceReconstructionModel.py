import torch
import torch.nn as nn
import numpy as np
import argparse
import nvdiffrast.torch as dr
import pickle
import torch.nn.functional as F

from tqdm import tqdm
from typing import Tuple
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from torch.optim.lr_scheduler import ExponentialLR

from flame.FLAME import FLAME, FLAMETex
from utils.transform import intrinsics_to_projection
from utils.utils import resize_long_side, scale_intrinsic_matrix
from utils.transform import rigid_transform_3d, rotation_matrix_to_axis_angle
from utils.loss import *


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

            face_verts = np.concatenate([masks["face"], masks["left_eyeball"], masks["right_eyeball"]])
            face_faces = []
            face_uvfaces = []

            for i, face, in enumerate(faces):
                for face_vert in face:
                    if int(face_vert) in face_verts:
                        face_faces.append(face)
                        face_uvfaces.append(uvfaces[i])
                        break

            self.face_uvfaces = torch.stack(face_uvfaces).int().to(self.device).contiguous()
            self.face_faces = torch.stack(face_faces).int().to(self.device).contiguous()

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
        self.intrinsics = None
        self.extrinsics = None

        self.coarse2fine_resolutions = list(
            map(lambda x: resize_long_side(
                orig_shape=orig_img_shape,
                dest_len=x
            ),
                face_model_config.coarse2fine_resolutions)
        )

    def set_transformation_matrices(
        self,
        extrinsic_matrices: torch.Tensor,
        intrinsic_matrices: torch.Tensor,
    ) -> None:
        # Get world to cam matrices
        self.world_to_cam = []
        self.extrinsics = []

        inverse = torch.Tensor(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, -1, 0],
             [0, 0, 0, 1]]
        )
        for extrinsic_matrix in extrinsic_matrices:
            self.world_to_cam.append((inverse @ torch.inverse(extrinsic_matrix)).t())
            self.extrinsics.append(extrinsic_matrix.t())

        self.world_to_cam = torch.stack(self.world_to_cam).float().to(self.device)
        self.extrinsics = torch.stack(self.extrinsics).float().to(self.device)

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

    # def set_initial_pose(
    #     self,
    #     input_landmarks: torch.Tensor
    # ):
    #     _, landmarks = self.face_model(
    #         shape_params=self.shape_coeffs,
    #         expression_params=self.exp_coeffs,
    #         pose_params=self.pose_coeffs
    #     )
    #
    #     input_landmarks = input_landmarks.cpu().numpy()[17:, :]
    #     landmarks = landmarks[0].detach().cpu().numpy()[17:, :]
    #
    #     not_nan_indices = ~(np.isnan(input_landmarks).any(axis=1))
    #     input_landmarks = input_landmarks[not_nan_indices]
    #     landmarks = landmarks[not_nan_indices]
    #
    #     r, t = rigid_transform_3d(input_landmarks.T, landmarks.T)
    #     r = rotation_matrix_to_axis_angle(r)
    #
    #     self.pose_coeffs = nn.Parameter(
    #         torch.from_numpy(np.concatenate([r, np.zeros(3)])[None, ...]).float().to(self.device))
    #
    #     self.transl_coeffs = nn.Parameter(
    #         torch.from_numpy(t).t().float().to(self.device)
    #     )

    def set_initial_pose(
            self,
            input_landmarks: torch.Tensor
    ):
        input_landmarks = input_landmarks[None, ...].float().to(self.device)
        input_landmarks = input_landmarks[:, 17:]

        not_nan_indices = ~(torch.isnan(input_landmarks).any(dim=-1))
        input_landmarks = input_landmarks[not_nan_indices]

        for resolution, lr, opt_steps, cam2ndc in zip(self.coarse2fine_resolutions, self.coarse2fine_lrs,
                                                      self.coarse2fine_opt_steps, self.cam_to_ndc):
            # Create optimizer
            optimizer = torch.optim.Adam(
                [{'params': [self.pose_coeffs, self.transl_coeffs]}],
                lr=lr,
            )
            scheduler = ExponentialLR(optimizer, gamma=0.999)

            for _ in tqdm(range(opt_steps)):
                _, landmarks_model = self.face_model(
                    shape_params=self.shape_coeffs,
                    expression_params=self.exp_coeffs,
                    pose_params=self.pose_coeffs,
                    transl=self.transl_coeffs
                )
                landmarks_model = landmarks_model[:, 17:][not_nan_indices][None, ...]
                landmarks_loss = landmark_distance(input_landmarks, landmarks_model, torch.ones(landmarks_model.shape[0], landmarks_model.shape[1]).to(self.device))

                loss = self.land_weight * landmarks_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

    def get_landmarks(self) -> torch.Tensor:
        _, landmarks = self.face_model(
            shape_params=self.shape_coeffs,
            expression_params=self.exp_coeffs,
            pose_params=self.pose_coeffs
        )
        return landmarks

    def _backproject_to_world_space(
        self,
        xys_cam: torch.Tensor,
        depths: torch.Tensor,
    ):
        # Convert to cam space
        xyzs_cam = depths * xys_cam

        # Convert to world space
        xyzs_cam = torch.concatenate([xyzs_cam, torch.ones(xyzs_cam.shape[0], xyzs_cam.shape[1], 1).to(self.device)],
                                     dim=-1)
        xyzs_world = torch.matmul(xyzs_cam, self.extrinsics)
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
        landmarks_world: torch.Tensor,
        world_to_cam: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vertices = torch.cat((vertices_world, torch.ones(vertices_world.shape[0], vertices_world.shape[1], 1).to(self.device)), dim=-1)
        landmarks = torch.cat((landmarks_world, torch.ones(landmarks_world.shape[0], landmarks_world.shape[1], 1).to(self.device)),
                              dim=-1)

        vertices_cam = torch.matmul(vertices, world_to_cam)
        landmarks_cam = torch.matmul(landmarks, world_to_cam)
        return vertices_cam, landmarks_cam

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
        resolution: tuple[int, int]
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

    def _render(
        self,
        verts_world: torch.Tensor,
        verts_cam: torch.Tensor,
        verts_ndc: torch.Tensor,
        albedos: torch.Tensor,
        resolution: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

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

        mask = (rast_out[..., 3] > 0).float().unsqueeze(1)
        return color, depth, mask

    def optimize(
        self,
        frame_features: dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        color, depth, input_color, input_depth = torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor()

        for resolution, lr, opt_steps, cam2ndc, intrinsic in zip(self.coarse2fine_resolutions, self.coarse2fine_lrs,
                                                                 self.coarse2fine_opt_steps, self.cam_to_ndc,
                                                                 self.intrinsics):
            inputs = torch.concatenate([frame_features["image"], frame_features["depth"], frame_features["normal"],
                                        frame_features["pixel_mask"]], dim=1)
            inputs_resized = F.interpolate(inputs, size=resolution).to(self.device)
            inputs_resized = inputs_resized.permute(0, 2, 3, 1)

            input_color = inputs_resized[..., :3].reshape(inputs_resized.shape[0], -1, 3)
            input_depth = inputs_resized[..., 3].reshape(inputs_resized.shape[0], -1, 1)
            input_normal = inputs_resized[..., 4:7].reshape(inputs_resized.shape[0], -1, 3)

            input_pixel_mask = inputs_resized[..., -1].reshape(inputs_resized.shape[0], -1)
            input_2d_landmarks = frame_features["predicted_landmark_2d"].float().to(self.device)
            input_2d_landmarks[:, :, 0] *= resolution[1] / self.orig_img_shape[1]
            input_2d_landmarks[:, :, 1] *= resolution[0] / self.orig_img_shape[0]

            # Get depth map with x, y coordinates
            x_indices, y_indices = np.meshgrid(np.arange(resolution[1]), np.arange(resolution[0]))
            x_indices = torch.from_numpy(x_indices.flatten()).to(self.device)[None, ...]
            x_indices = torch.repeat_interleave(x_indices, inputs_resized.shape[0], 0)[..., None]
            y_indices = torch.from_numpy(y_indices.flatten()).to(self.device)[None, ...]
            y_indices = torch.repeat_interleave(y_indices, inputs_resized.shape[0], 0)[..., None]
            xy_coordinates = torch.concatenate([x_indices, y_indices], dim=-1)

            # Create optimizer
            optimizer = torch.optim.Adam(
                [{'params': [self.shape_coeffs, self.exp_coeffs, self.pose_coeffs, self.tex_coeffs,
                             self.transl_coeffs, self.light_coeffs, self.albedo]}],
                lr=lr,
            )
            scheduler = ExponentialLR(optimizer, gamma=0.999)

            for _ in tqdm(range(opt_steps)):
                # Get vertices in cam space and compute vertices attributes
                vertices_world, landmarks_world = self.face_model(
                    shape_params=self.shape_coeffs,
                    expression_params=self.exp_coeffs,
                    pose_params=self.pose_coeffs,
                    transl=self.transl_coeffs
                )

                vertices_cam, landmarks_cam = self._project_to_cam_space(vertices_world, landmarks_world, self.world_to_cam)
                albedos = (self.texture_model(self.tex_coeffs) / 255.0).permute(0, 2, 3, 1).contiguous()

                vertices_ndc = self._project_to_ndc_space(vertices_cam, cam2ndc)
                landmarks_ndc = self._project_to_ndc_space(landmarks_cam, cam2ndc)
                landmarks_ndc[:, :, 1] = -landmarks_ndc[:, :, 1]
                landmarks_screen = self._project_to_image_space(landmarks_ndc, resolution)

                color, depth, pixel_mask = self._render(vertices_world, vertices_cam, vertices_ndc, albedos, resolution)

                # normal image
                zy, zx = torch.gradient(depth, dim=(1, 2))
                normal = torch.concatenate([-zx, -zy, torch.ones_like(depth)], dim=-1)
                norm = torch.norm(normal, dim=-1, keepdim=True)
                normal = normal / norm

                color = color.reshape(color.shape[0], -1, 3)
                depth = -depth.reshape(depth.shape[0], -1, 1)
                normal = normal.permute(0, 2, 3, 1).reshape(normal.shape[0], -1, 3)

                xys_homo = torch.concatenate(
                    [xy_coordinates, torch.ones(xy_coordinates.shape[0], xy_coordinates.shape[1], 1).to(self.device)],
                    dim=-1)
                xys_cam = torch.matmul(xys_homo, intrinsic)

                pixels_world_input = self._backproject_to_world_space(xys_cam, input_depth)
                pixels_world_rendered = self._backproject_to_world_space(xys_cam, depth)

                normals_world_input = self._backproject_to_world_space_normal(input_normal)
                normals_world_rendered = self._backproject_to_world_space_normal(normal)

                # Compute losses
                landmarks_mask = torch.ones(landmarks_screen.shape[:2]).to(self.device)
                pixel_mask = pixel_mask.reshape(pixel_mask.shape[0], -1)
                pixel_mask *= input_pixel_mask
                num_valid_pixels = (pixel_mask > 0).sum(-1)

                landmarks_loss = landmark_distance(input_2d_landmarks, landmarks_screen, landmarks_mask)
                rgb_loss = pixel2pixel_distance(input_color, color, pixel_mask, num_valid_pixels)
                p2point_loss = pixel2pixel_distance(pixels_world_input, pixels_world_rendered, pixel_mask,
                                                    num_valid_pixels)
                p2plane_loss = point2plane_distance(pixels_world_input, normals_world_input, pixels_world_rendered,
                                                    normals_world_rendered, pixel_mask, num_valid_pixels)

                loss = self.land_weight * landmarks_loss + \
                       self.p2point_weight * p2point_loss + \
                       self.p2plane_weight * p2plane_loss + \
                       self.rgb_weight * rgb_loss + \
                       self.shape_reg_weight * self.shape_coeffs.norm(p=2, dim=1) + \
                       self.exp_reg_weight * self.exp_coeffs.norm(p=2, dim=1) + \
                       self.tex_reg_weight * self.tex_coeffs.norm(p=2, dim=1)

                print("landmark: ", self.land_weight * landmarks_loss)
                print("p2point: ", self.p2point_weight * p2point_loss)
                print("p2plane: ", self.p2plane_weight * p2plane_loss)
                print("rgb: ", self.rgb_weight * rgb_loss)
                print("shape reg: ", self.shape_reg_weight * self.shape_coeffs.norm(p=2, dim=1))
                print("exp reg: ", self.exp_reg_weight * self.exp_coeffs.norm(p=2, dim=1))
                print("tex reg: ", self.tex_reg_weight * self.tex_coeffs.norm(p=2, dim=1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            import pyvista as pv
            pixels_world_rendered_ = (pixels_world_rendered[pixels_world_rendered[..., -1] != 0][
                (pixel_mask > 0).flatten()]).detach().cpu().numpy()
            pixels_world_input_ = (
            pixels_world_input[pixels_world_input[..., -1] != 0][(pixel_mask > 0).flatten()]).detach().cpu().numpy()

            plotter = pv.Plotter()
            plotter.camera_position = 'xy'
            plotter.add_mesh(pv.PolyData(pixels_world_rendered_), color='green')
            plotter.add_mesh(pv.PolyData(pixels_world_input_), color='red')
            plotter.add_mesh(pv.PolyData(vertices_world[0].detach().cpu().numpy()), color='blue')
            plotter.show()

        color = color.reshape(color.shape[0], *self.coarse2fine_resolutions[-1], 3)
        depth = depth.reshape(depth.shape[0], *self.coarse2fine_resolutions[-1], 1)
        input_color = input_color.reshape(input_color.shape[0], *self.coarse2fine_resolutions[-1], 3)
        input_depth = input_depth.reshape(input_depth.shape[0], *self.coarse2fine_resolutions[-1], 1)
        return color, depth, input_color, input_depth
