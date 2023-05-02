import numpy as np
import pyvista as pv

def visualize_3d_scan_and_landmarks(points: np.ndarray, landmarks: np.ndarray):
    pv_points = pv.PolyData(points)
    pv_landmarks = pv.PolyData(landmarks)
    plotter = pv.Plotter()
    plotter.add_mesh(pv_points, color='green', opacity=0.1, point_size=1)
    plotter.add_mesh(pv_landmarks, color='red', point_size=5)
    plotter.show()


def visualize_3d_face_model(points: np.ndarray, faces: np.ndarray):
    mesh = pv.PolyData(points,
                       np.concatenate((np.ones((faces.shape[0], 1)) * 3, faces), axis=-1).reshape(-1).astype(np.int64))
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='green')
    plotter.add_axes(line_width=5, labels_off=False)
    plotter.show()


def visualize_3d_scan_and_3d_face_model(points_scan: np.ndarray,
                                        points_3d_face: np.ndarray,
                                        faces_3d_face: np.ndarray):
    pv_scan_points = pv.PolyData(points_scan)
    pv_3d_mesh = pv.PolyData(points_3d_face,
                             np.concatenate((np.ones((faces_3d_face.shape[0], 1)) * 3,
                                             faces_3d_face), axis=-1).reshape(-1).astype(np.int64))
    plotter = pv.Plotter()
    plotter.add_mesh(pv_scan_points, color='red', opacity=0.1, point_size=1)
    plotter.add_mesh(pv_3d_mesh, color='green')
    plotter.add_axes(line_width=5, labels_off=False)
    plotter.show()
