import numpy as np
import pyvista as pv
from dreifus.pyvista import add_floor, add_coordinate_axes, add_camera_frustum, Pose, Intrinsics

"""
    Script used to plot camera frustum in 3D
"""

p = pv.Plotter(notebook=False)
add_floor(p, square_size=0.1, max_distance=1)
add_coordinate_axes(p, scale=0.1, draw_labels=False)

k_2 = np.load("../data/subject_0/222200037/intrinsics.npy")
k_2 = Intrinsics(k_2)
E_2 = np.load("../data/subject_0/222200037/extrinsics.npy")
E_2 = np.linalg.inv(E_2)
E_2_p = Pose(E_2)

base_transf_2 = np.eye(4)
base_transf_2[:3] = np.genfromtxt("../data/subject_0_tracked_mesh/00000_transform.txt", max_rows=3)
base_transf_2[:3, 3] /= 1000
base_transf_2_p = Pose(base_transf_2)

E_2_t = E_2@base_transf_2
E_2_t_p = Pose(E_2_t)

add_camera_frustum(p, E_2_p, k_2, color='yellow')
add_camera_frustum(p, E_2_t_p, k_2, color='orange')
add_camera_frustum(p, base_transf_2_p, k_2, color='red')

axes_actor = p.add_axes()
axes_actor.SetXAxisLabelText("X")
axes_actor.SetYAxisLabelText("Y")
axes_actor.SetZAxisLabelText("Z")
p.show()