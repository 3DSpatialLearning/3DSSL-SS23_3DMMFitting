import numpy as np
import cv2
import torch
from scipy.spatial import distance


def rotation_matrix_to_axis_angle(rotation: np.ndarray) -> np.ndarray:
    assert rotation.shape == (3, 3), "Rotation matrix must be 3x3"
    axis_angle, _ = cv2.Rodrigues(rotation)
    return axis_angle.squeeze()


def axis_angle_to_rotation_matrix(axis_angle: np.ndarray) -> np.ndarray:
    assert axis_angle.shape == (3,), "Axis angle must be 3 dimensional"
    rotation, _ = cv2.Rodrigues(axis_angle)
    return rotation.squeeze()


def intrinsics_to_projection(intrinsics: torch.Tensor, resolution: tuple[int, int], znear: float = .01,
                             zfar: float = 10.) -> torch.Tensor:
    """
    Converts the given intrinsics matrix to a OpenGL projection matrix.
    :param intrinsics: 3x3
    :return: 4x4 OpenGL projection matrix
    """
    assert intrinsics.shape == (3, 3)
    fx, fy = intrinsics[0, 0] * (resolution[1] / (intrinsics[0, 2] * 2)), intrinsics[1, 1] * (
                resolution[0] / (intrinsics[1, 2] * 2))
    cx, cy = resolution[1] / 2, resolution[0] / 2
    sx = intrinsics[0, 1]
    w, h = resolution[1], resolution[0]

    projection_matrix = torch.Tensor(
        [[2 * fx / w, -2 * sx / w, (w - 2 * cx) / w, 0],
         [0, 2 * fy / h, (-h + 2 * cy) / h, 0],
         [0, 0, -(zfar + znear) / (zfar - znear), -2 * (zfar * znear) / (zfar - znear)],
         [0, 0, -1, 0]]
    )
    return projection_matrix


def filter_outliers_landmarks(landmark: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Filters out the outliers from the landmarks by setting corresponding outlier positions to nan.
    :param landmark: Nx3
    :return: Nx3
    """
    dist_matrix = distance.cdist(landmark, landmark)
    np.fill_diagonal(dist_matrix, np.inf)
    min_distances = np.min(dist_matrix, axis=1)
    landmark[min_distances == 0] = np.nan
    landmark[min_distances > threshold] = np.nan
    return landmark


def get_coordinates_from_depth_map_by_threshold(depth_map: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Returns the coordinates of the pixels in the depth map that are above the given threshold.
    :param depth_map HxW
    :param threshold float
    :return: Nx2 array of pixel coordinates
    """
    ys, xs = np.where(depth_map > threshold)
    pixel_coordinates = np.column_stack((xs, ys))
    return pixel_coordinates


def backproject_points(points: np.ndarray, depth_map: np.ndarray, K: np.ndarray, E: np.ndarray) -> np.ndarray:
    """
    Backprojects a set of 2D pixels into 2D points using the given camera intrinsics and extrinsics.
    :param points: 2D points Nx2
    :param depth_map: Depth map of the image HxW
    :param intrinsics: Camera intrinsics 3x3
    :param extrinsics: Camera extrinsics 4x4
    :return: Nx3 points in world space
    """
    assert points.shape[1] == 2
    assert K.shape == (3, 3)
    assert E.shape == (4, 4)
    depths = depth_map[points[:, 1], points[:, 0]].reshape(-1, 1)
    # screen space to cam space
    points_homo = np.hstack((points, np.ones((len(points), 1)))).T
    cam_coords = depths * (np.linalg.inv(K) @ points_homo).T

    # camera space to world space
    cam_coords_homo = np.hstack((cam_coords, np.ones((len(points), 1))))
    world_coords_homo = E @ cam_coords_homo.T
    world_coords = world_coords_homo[:3, :].T

    return world_coords


def rigid_transform_3d(source_points: np.ndarray, dest_points: np.ndarray):
    # Taken from https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
    assert source_points.shape == dest_points.shape

    num_rows, num_cols = source_points.shape
    if num_rows != 3:
        raise Exception(f"matrix source_points is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = dest_points.shape
    if num_rows != 3:
        raise Exception(f"matrix dest_points is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_source_points = np.mean(source_points, axis=1)
    centroid_dest_points = np.mean(dest_points, axis=1)

    # ensure centroids are 3x1
    centroid_source_points = centroid_source_points.reshape(-1, 1)
    centroid_dest_points = centroid_dest_points.reshape(-1, 1)

    # subtract mean
    source_points_m = source_points - centroid_source_points
    dest_points_m = dest_points - centroid_dest_points

    H = source_points_m @ np.transpose(dest_points_m)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_source_points + centroid_dest_points

    return R, t
