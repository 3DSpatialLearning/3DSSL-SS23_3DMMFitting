import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from utils.data import load_data
from utils.fitting import fit_flame_model_to_input_point_cloud
from utils.visualization import visualize_3d_scan_and_3d_face_model

LANDMARKS_FILE = "./data/toy_task/team1_landmarks.npy"
POINTS_FILE = "./data/toy_task/team1_points.npy"
NORMALS_FILE = "./data/toy_task/team1_normals.npy"


if __name__ == '__main__':
    scale = 1./1000.
    steps = 100
    lr = 0.01
    wd = 1e-6
    input_data = load_data(scale=scale,
                           landmarks_path=LANDMARKS_FILE,
                           points_path=POINTS_FILE,
                           normals_path=NORMALS_FILE)
    optimal_parameters = fit_flame_model_to_input_point_cloud(input_data=input_data,
                                                              steps=steps,
                                                              lr=lr,
                                                              wd=wd)
