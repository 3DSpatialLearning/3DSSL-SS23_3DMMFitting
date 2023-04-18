import numpy as np
import pyvista as pv

LANDMARKS_FILE = "./data/team1_landmarks.npy"
POINTS_FILE = "./data/team1_points.npy"
NORMALS_FILE = "./data/team1_normals.npy"

def visualize_points():
    landmarks = np.load(LANDMARKS_FILE)
    points = np.load(POINTS_FILE)
    pv_landmarks = pv.PolyData(landmarks)
    pv_points = pv.PolyData(points)
    plotter = pv.Plotter()
    plotter.add_mesh(pv_landmarks, color='red', point_size=5)
    plotter.add_mesh(pv_points, color='green', opacity=0.1, point_size=1)
    plotter.show()

if __name__ == '__main__':
    visualize_points()
