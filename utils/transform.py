import numpy as np

def backproject_points(points: np.ndarray, depth_map: np.ndarray, K: np.ndarray, E: np.ndarray) -> np.ndarray:
    """
    Backprojects a set of 2D pixels into 2D points using the given camera intrinsics and extrinsics.
    :param points: 2D points Nx2
    :param intrinsics: Camera intrinsics 3x3
    :param extrinsics: Camera extrinsics 4x4
    :return: 3D points
    """
    assert points.shape[1] == 2
    assert K.shape == (3, 3)
    assert E.shape == (4, 4)

    depths = depth_map[points[:, 1], points[:, 0]].reshape(-1, 1)

    # screen space to cam space
    points_homo = np.hstack((points, np.ones((len(points), 1)))).T
    cam_coords = depths * (np.linalg.inv(K)@points_homo).T

    # camera space to world space
    cam_coords_homo = np.hstack((cam_coords, np.ones((len(points), 1))))
    world_coords_homo = E@cam_coords_homo.T
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
