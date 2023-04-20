import numpy as np
import pyvista as pv

def visualize_points(data: dict[str, np.ndarray]):
    pv_landmarks = pv.PolyData(data["landmarks"])
    pv_points = pv.PolyData(data["points"])
    plotter = pv.Plotter()
    plotter.add_mesh(pv_landmarks, color='red', point_size=5)
    plotter.add_mesh(pv_points, color='green', opacity=0.1, point_size=1)
    plotter.show()