import numpy as np
import pyvista as pv


def visualize_points(
    landmarks_file_path: str,
    points_file_path: str
):
    landmarks = np.load(landmarks_file_path)
    points = np.load(points_file_path)
    pv_landmarks = pv.PolyData(landmarks)
    pv_points = pv.PolyData(points)
    plotter = pv.Plotter()
    plotter.add_mesh(pv_landmarks, color='red', point_size=5)
    plotter.add_mesh(pv_points, color='green', opacity=0.1, point_size=1)
    plotter.show()


if __name__ == "__main__":
    visualize_points(
        landmarks_file_path="./data/toy_task/team1_landmarks.npy",
        points_file_path="./data/toy_task/team1_points.npy"
    )
