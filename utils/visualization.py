import numpy as np
import pyvista as pv

def visualize_point_clouds(point_clouds: list[np.ndarray], colors: list[str] = None):
    plotter = pv.Plotter()
    for i, points in enumerate(point_clouds):
        if colors is not None:
            color = colors[i]
        plotter.add_mesh(pv.PolyData(points), color=color, point_size=3)
    plotter.show()

def visualize_3d_face_model(points: np.ndarray, faces: np.ndarray, screenshot: bool = False,
                                        screenshot_path: str = None):
    mesh = pv.PolyData(points,
                       np.concatenate((np.ones((faces.shape[0], 1)) * 3, faces), axis=-1).reshape(-1).astype(np.int64))
    plotter = pv.Plotter(off_screen=screenshot)
    plotter.camera_position = 'xy'
    plotter.add_mesh(mesh, color='green')
    plotter.add_axes(line_width=5, labels_off=False)
    if screenshot:
        plotter.screenshot(screenshot_path)
    else:
        plotter.show()


def visualize_3d_scan_and_3d_face_model(points_scan: np.ndarray,
                                        points_3d_face: np.ndarray,
                                        faces_3d_face: np.ndarray,
                                        predicted_landmarks_3d: np.ndarray = None,
                                        screenshot_path: str = None):
    pv_scan_points = pv.PolyData(points_scan)
    pv_3d_mesh = pv.PolyData(points_3d_face,
                             np.concatenate((np.ones((faces_3d_face.shape[0], 1)) * 3,
                                             faces_3d_face), axis=-1).reshape(-1).astype(np.int64))
    pv_predicted_landmarks_3d = predicted_landmarks_3d

    if screenshot_path is not None:
        plotter = pv.Plotter(shape=(1, 3), off_screen=True)
        plotter.set_background('white')
    else:
        plotter = pv.Plotter(shape=(1, 3))

    plotter.subplot(0, 0)
    plotter.add_mesh(pv_scan_points, color='red', opacity=0.2, point_size=1)
    plotter.add_mesh(pv_predicted_landmarks_3d, color='blue', point_size=7, style='points')
    plotter.add_text("Face Scan")
    plotter.add_axes(line_width=5, labels_off=False)

    plotter.subplot(0, 1)
    plotter.add_mesh(pv_3d_mesh, color='green')
    plotter.add_text("Flame model")
    plotter.add_axes(line_width=5, labels_off=False)

    plotter.subplot(0, 2)
    plotter.add_mesh(pv_scan_points, color='red', opacity=0.5, point_size=1)
    plotter.add_mesh(pv_3d_mesh, color='green')
    plotter.add_text("Combined View")
    plotter.add_axes(line_width=5, labels_off=False)

    plotter.link_views()
    plotter.camera_position = 'xy'

    if screenshot_path is not None:
        plotter.screenshot(screenshot_path)
    else:
        plotter.show()

def visualize_landmark_alignment(input_landmarks_3d: np.ndarray, flame_landmarks_3d_before_alignment: np.ndarray, flame_landmarks_3d_after_alignment: np.ndarray, screenshot_path: str = None):
    if screenshot_path is not None:
        plotter = pv.Plotter(shape=(1,2), off_screen=True)
        plotter.set_background('white')
    else:
        plotter = pv.Plotter(shape=(1,2))

    error = 0
    plotter.subplot(0, 0)
    plotter.add_mesh(input_landmarks_3d, color='blue', point_size=5, style='points')
    plotter.add_mesh(flame_landmarks_3d_before_alignment, color='red', point_size=5, style='points')
    for point_a, point_b in zip(input_landmarks_3d, flame_landmarks_3d_before_alignment):
        error += np.linalg.norm(point_a - point_b)
        line = pv.Line(point_a, point_b)
        plotter.add_mesh(line, color='black', line_width=2, opacity=0.25)
    plotter.camera_position = 'xy'
    plotter.add_text("Before Rigid alignment")
    plotter.add_text("Error: {:.4f}".format(error), position='lower_left', font_size=9)
    plotter.add_axes(line_width=5, labels_off=False)

    error = 0
    plotter.subplot(0, 1)
    plotter.add_mesh(input_landmarks_3d, color='blue', point_size=5, style='points')
    plotter.add_mesh(flame_landmarks_3d_after_alignment, color='red', point_size=5, style='points')
    for point_a, point_b in zip(input_landmarks_3d, flame_landmarks_3d_after_alignment):
        error += np.linalg.norm(point_a - point_b)
        line = pv.Line(point_a, point_b)
        plotter.add_mesh(line, color='black', line_width=2, opacity=0.25)
    plotter.camera_position = 'xy'
    plotter.add_text("After Rigid alignment")
    plotter.add_text("Error: {:.4f}".format(error), position='lower_left', font_size=9)
    plotter.add_axes(line_width=5, labels_off=False)

    if screenshot_path is not None:
        plotter.screenshot(screenshot_path)
    else:
        plotter.show()
