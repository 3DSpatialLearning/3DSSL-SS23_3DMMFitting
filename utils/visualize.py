import numpy as np
import pyvista as pv

def display_points(
    title: str = "Visualization", 
    points: np.ndarray = None,
    landmarks: np.ndarray = None,
    normals: np.ndarray = None,
    normal_length: float = 1
):
    plotter = pv.Plotter()

    if points is not None:
        plotter.add_mesh(pv.PolyData(points), color='green', point_size=1, opacity=0.5)

    if landmarks is not None:
        plotter.add_mesh(pv.PolyData(landmarks), color='red', point_size=5)

    if normals is not None:
        # calculate the normals line to display
        magnitudes = np.sqrt(np.sum(normals**2, axis=1))
        normals_unit = (normals.T / magnitudes).T
        normals_end = points + normals_unit*normal_length
        normal_lines = np.stack((points[:, :], normals_end[:, :]), axis=1)
        normal_lines = normal_lines.reshape(-1, normal_lines.shape[-1])

        plotter.add_lines(normal_lines, color='blue', width=1)
    
    # show
    plotter.show(title)