import numpy as np
import torch
from flame.config import get_config
from flame.FLAME import FLAME
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from pytorch3d.loss.chamfer import chamfer_distance
from tqdm import tqdm

import pyvista as pv


def rigid_transform_3D(A, B):
    # Taken from https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


if __name__ == "__main__":
    scale_factor = 1000

    config = get_config()
    flame_model = FLAME(config)

    landmarks = np.load("./data/toy_task/team1_landmarks.npy") / scale_factor
    points = np.load("./data/toy_task/team1_points.npy") / scale_factor
    normals = np.load("./data/toy_task/team1_normals.npy")

    # Get the coordinate of Flame landmarks
    shape = torch.zeros(1, 100).float()
    exp = torch.zeros(1, 50).float()
    pose = torch.zeros(1, 6).float()

    _, flame_landmarks = flame_model(shape, exp, pose)
    flame_landmarks = flame_landmarks.squeeze()

    landmarks_input = landmarks[15:24].transpose()
    landmarks_source = flame_landmarks[10:19].numpy().transpose()

    # Find the rigid transformation
    R, t = rigid_transform_3D(landmarks_input, landmarks_source)

    landmarks = (R @ landmarks.transpose() + t).transpose()
    points = (R @ points.transpose() + t).transpose()

    # Optimize for parameters
    steps = 1000
    lr = 0.01

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    landmarks = torch.tensor(landmarks).float().to(device).unsqueeze(0)

    flame_model.to(device)

    shape = torch.zeros(1, 100, requires_grad=True, device=device)
    exp = torch.zeros(1, 50, requires_grad=True, device=device)
    pose = torch.zeros(1, 6, requires_grad=True, device=device)

    optimizer = torch.optim.Adam(
        [shape, exp, pose],
        lr=lr,
    )
    scheduler = ExponentialLR(optimizer, gamma=0.999)

    loss_fn = chamfer_distance

    pbar = tqdm(range(steps))

    for step in pbar:
        optimizer.zero_grad()
        vertices, predicted_landmarks = flame_model(shape_params=shape, expression_params=exp, pose_params=pose)
        loss, _ = loss_fn(landmarks, predicted_landmarks)
        loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.set_postfix(loss=loss.item())

    vertices = vertices.detach().cpu().squeeze().numpy()

    pv_points = pv.PolyData(vertices)
    plotter = pv.Plotter()
    plotter.add_mesh(pv_points, color='green', point_size=1)
    plotter.show()
