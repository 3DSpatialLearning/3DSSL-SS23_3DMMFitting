import numpy as np
import pyvista as pv
from dreifus.pyvista import add_floor, add_coordinate_axes, add_camera_frustum, Pose, Intrinsics

p = pv.Plotter(notebook=False)
add_floor(p, square_size=0.1, max_distance=1)
add_coordinate_axes(p, scale=0.1, draw_labels=False)

k_1 = np.array([
    [7727.212, 0.0, 830.3808],
    [0.0, 7727.479, 937.41406],
    [0.0, 0.0, 1.0]
])
E_1 = np.array([
    [0.9648338, 0.025369735, 0.2616336, -296.94205],
    [0.022339592, 0.9838167, -0.17777976, 182.69637],
    [-0.26190972, 0.17737271, 0.94865286, 85.425545],
    [0, 0, 0, 1]
])
E_1[:3, 3] /= 1000
E_1_p = Pose(E_1)

k_1 = Intrinsics(k_1)
base_transf_1 = np.eye(4)
base_transf_1[:3] = np.genfromtxt("../models/mvp/experiments/example/data/geom/tracked_mesh/E001_Neutral_Eyes_Open/000102_transform.txt", max_rows=3)
base_transf_1[:3, 3] /= 1000
base_transf_1_p = Pose(base_transf_1)

E_1_t = E_1@base_transf_1
E_1_t_p = Pose(E_1_t)

k_2 = np.load("../data/subject_0/222200037/intrinsics.npy")
k_2 = Intrinsics(k_2)
E_2 = np.load("../data/subject_0/222200037/extrinsics.npy")
E_2 = np.linalg.inv(E_2)
E_2[1:3, :3] *= -1
E_2_p = Pose(E_2)

base_transf_2 = np.eye(4)
base_transf_2[:3] = np.genfromtxt("../data/subject_0_tracked_mesh/00000_transform.txt", max_rows=3)
base_transf_2[:3, 3] /= 1000
# base_transf_2[1:3] *= -1
base_transf_2_p = Pose(base_transf_2)

E_2_t = E_2@base_transf_2
E_2_t_p = Pose(E_2_t)

add_camera_frustum(p, E_1_p, k_1, color='yellow')
add_camera_frustum(p, E_1_t_p, k_1, color='orange')
add_camera_frustum(p, base_transf_1_p, k_1, color='red')

print(E_1_t)
# add_camera_frustum(p, E_2_p, k_2, color='yellow')
# add_camera_frustum(p, E_2_t_p, k_2, color='orange')
# add_camera_frustum(p, base_transf_2_p, k_2, color='red')


axes_actor = p.add_axes()
axes_actor.SetXAxisLabelText("X")
axes_actor.SetYAxisLabelText("Y")
axes_actor.SetZAxisLabelText("Z")
p.show()