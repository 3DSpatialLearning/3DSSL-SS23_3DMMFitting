from utils.data import load_data
from utils.fitting import fit_flame_model_to_input_point_cloud

LANDMARKS_FILE = "./data/toy_task/single_frame_point_cloud_fitting/team1_landmarks.npy"
POINTS_FILE = "./data/toy_task/single_frame_point_cloud_fitting/team1_points.npy"
NORMALS_FILE = "./data/toy_task/single_frame_point_cloud_fitting/team1_normals.npy"

if __name__ == '__main__':
    scale = 1./1000.
    steps = 200
    lr = 0.01
    input_data = load_data(scale=scale,
                           landmarks_path=LANDMARKS_FILE,
                           points_path=POINTS_FILE,
                           normals_path=NORMALS_FILE)
    optimal_parameters = fit_flame_model_to_input_point_cloud(input_data=input_data,
                                                              steps=steps,
                                                              lr=lr)
