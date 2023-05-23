import torch.nn as nn
import torch
import numpy as np
from flame.config import get_config
from flame.FLAME import FLAME
from utils.data import load_data
from utils.fitting import fit_flame_model_to_input_point_cloud
import os

main_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(str(main_dir))

LANDMARKS_FILE = "../data/toy_task/single_frame_point_cloud_fitting/team1_landmarks.npy"
POINTS_FILE = "../data/toy_task/single_frame_point_cloud_fitting/team1_points.npy"
NORMALS_FILE = "../data/toy_task/single_frame_point_cloud_fitting/team1_normals.npy"

if __name__ == '__main__':
    scale = 1./1000.
    steps = 200
    lr = 0.01
    input_data = load_data(scale=scale,
                           landmarks_path=LANDMARKS_FILE,
                           points_path=POINTS_FILE,
                           normals_path=NORMALS_FILE)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = get_config()
    flame_model = FLAME(config)
    flame_model_faces = torch.from_numpy(flame_model.faces.astype(np.int32)).unsqueeze(0).to(device)
    flame_model.to(device)
    shape = nn.Parameter(torch.zeros(1, config.shape_params).float().to(device))
    exp = nn.Parameter(torch.zeros(1, config.expression_params).float().to(device))
    pose = nn.Parameter(torch.zeros(1, config.pose_params).float().to(device))
    optimal_parameters = fit_flame_model_to_input_point_cloud(input_data=input_data,
                                                              steps=steps,
                                                              lr=lr)
