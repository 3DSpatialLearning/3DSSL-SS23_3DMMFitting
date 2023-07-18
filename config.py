import argparse
import torch
import json

parser = argparse.ArgumentParser(description='FLAME fitting config')

#################### General settings ####################
parser.add_argument(
    '--experiment_name',
    type=str,
    default='test',
    help='experiment name',
)

parser.add_argument(
    '--output_frame_rate',
    type=int,
    default=24,
    help='output frame rate'
)

parser.add_argument(
    '--output_video_name',
    type=str,
    default='fitting.mp4',
    help='output video name'
)

parser.add_argument(
    '--workdir',
    type=str,
    default='output/',
    help='workdir'
)

parser.add_argument(
    '--draw_landmarks',
    type=bool,
    default=True,
    help='If true draw landmarks on the output video'
)

#################### FLAME args ####################

parser.add_argument(
    '--flame_model_path',
    type=str,
    default='data/flame_model/generic_model.pkl',
    help='flame model path'
)

parser.add_argument(
    '--mediapipe_landmark_embedding_path',
    type=str,
    default='data/flame_model/mediapipe_landmark_embedding.npz',
    help='Mediapipe landmark embedding path for FLAME'
)

parser.add_argument(
    '--static_landmark_embedding_path',
    type=str,
    default='data/flame_model/flame_static_embedding.pkl',
    help='Static landmark embeddings path for FLAME'
)

parser.add_argument(
    '--dynamic_landmark_embedding_path',
    type=str,
    default='data/flame_model/flame_dynamic_embedding.npy',
    help='Dynamic contour embedding path for FLAME'
)

parser.add_argument(
    '--tex_space_path',
    type=str,
    default='data/flame_model/FLAME_albedo_from_BFM.npz',
    help='Texture space path for FLAME'
)

parser.add_argument(
    '--head_template_mesh_path',
    type=str,
    default='data/flame_model/head_template_mesh.obj',
    help='head template mesh for FLAME'
)

parser.add_argument(
    '--flame_masks_path',
    type=str,
    default='data/flame_model/FLAME_masks.pkl',
    help='flame masks'
)
# FLAME hyper-parameters

parser.add_argument(
    '--shape_params',
    type=int,
    default=300,
    help='the number of shape parameters'
)

parser.add_argument(
    '--expression_params',
    type=int,
    default=100,
    help='the number of expression parameters'
)

parser.add_argument(
    '--pose_params',
    type=int,
    default=6,
    help='the number of pose parameters'
)

parser.add_argument(
    '--neck_pose_params',
    type=int,
    default=3,
    help='the number of neck pose parameters'
)

parser.add_argument(
    '--eye_pose_params',
    type=int,
    default=6,
    help='the number of eye pose parameters'
)

parser.add_argument(
    '--tex_params',
    type=int,
    default=100,
    help='the number of texture parameters'
)

parser.add_argument(
    '--use_face_contour',
    default=True,
    type=bool,
    help='If true apply the landmark loss on also on the face contour.'
)

parser.add_argument(
    '--use_3D_translation',
    default=True,  # Flase for RingNet project
    type=bool,
    help='If true apply the landmark loss on also on the face contour.'
)

parser.add_argument(
    '--batch_size',
    type=int,
    default=1,
    help='Training batch size.'
)

####################### General  #######################

parser.add_argument(
    '--cam_data_dir',
    type=str,
    default="data/subject_0",
    help='Directory containing the camera data'
)

# "[222200036, 222200037, 222200038, 222200039, 222200041, 222200042, 222200044, 222200045, 222200046, 222200047, 222200048, 222200049]"

parser.add_argument(
    '--depth_camera_ids',
    type=json.loads,
    default="[222200036, 222200037, 222200038, 222200039, 222200041, 222200042, 222200044, 222200045, 222200046, 222200047, 222200048, 222200049]",
    help='Id of the cameras that are used for the depth fitting. Note, depth camera'
         'ids must be a subset of the rgb camera ids.'
)

parser.add_argument(
    '--rgb_camera_ids',
    type=json.loads,
    default="[222200036, 222200037, 222200038, 222200039, 222200041, 222200042, 222200044, 222200045, 222200046, 222200047, 222200048, 222200049]",
    help='Id of the cameras that are used for the rgb fitting'
)

parser.add_argument(
    '--landmark_camera_id',
    type=json.loads,
    default="[222200037]",
    help='Id of the camera that are used for the landmark fitting. It can only be one camera.'
)

parser.add_argument(
    '--frontal_camera_id',
    type=json.loads,
    default="[222200037]",
    help='Id of the frontal camera',
)

parser.add_argument(
    '--scan_2_mesh_camera_ids',
    type=json.loads,
    default="[222200036, 222200037, 222200049]",
    help='Id of the cameras that are used to compute the scan to mesh distance',
)

parser.add_argument(
    '--device',
    type=str,
    default=None,
    help='Device to run the operations'
)

parser.add_argument(
    '--dlib_face_predictor_path',
    type=str,
    default="data/checkpoints/shape_predictor_68_face_landmarks.dat",
    help='Path to dlib face predictor checkpoint'
)

parser.add_argument(
    '--deca_output_path',
    type=str,
    default="data/deca/subject_0",
    help='Path to deca output'
)

####################### Fitting #######################

parser.add_argument(
    '--num_samples_flame',
    type=int,
    default=10000,
    help='Num of points to sample from Flame mesh'
)

parser.add_argument(
    '--num_samples_scan',
    type=int,
    default=20000,
    help='Num of points to sample from scan mesh'
)

parser.add_argument(
    '--coarse2fine_resolutions',
    type=list,
    default=[224, 448, 760],
    help='resolutions used for the coarse to fine optimization strategy'
)

parser.add_argument(
    '--coarse2fine_lrs_first_frame',
    type=list,
    default=[1e-2, 2e-3, 3e-4],
    help='learning rate associated to every level for the first frame'
)

parser.add_argument(
    '--coarse2fine_lrs_next_frames',
    type=list,
    default=[1e-2, 5e-3, 1e-3],
    help='learning rate associated to every level for the first frame'
)

parser.add_argument(
    '--coarse2fine_opt_steps_first_frame',
    type=list,
    default=[300, 200, 100],
    help='number of optimization steps associated to every level for the first frame'
)

parser.add_argument(
    '--coarse2fine_opt_steps_next_frames',
    type=list,
    default=[50, 30, 20],
    help='number of optimization steps associated to every level for the next frames'
)

parser.add_argument(
    '--scan_to_mesh_weight',
    type=float,
    default=1.0,
    help='Scan to mesh term weight'
)

parser.add_argument(
    '--scan_to_face_weight',
    type=float,
    default=1e-2,
    help='Scan to face term weight'
)

parser.add_argument(
    '--landmarks_68_weight',
    type=float,
    default=0.125,
    help='Landmark term weight'
)

parser.add_argument(
    '--shape_regularization_weight',
    type=float,
    default=0.01,
    help='Shape regularization weight'
)

parser.add_argument(
    '--exp_regularization_weight',
    type=float,
    default=0.01,
    help='Expression regularization weight'
)

parser.add_argument(
    '--tex_regularization_weight',
    type=float,
    default=0.025,
    help='Texture regularization weight'
)

parser.add_argument(
    '--rgb_weight',
    type=float,
    default=20,
    help='color loss weight'
)

parser.add_argument(
    '--point2point_weight',
    type=float,
    default=20000,
    help='point to point loss weight'
)

parser.add_argument(
    '--point2plane_weight',
    type=float,
    default=30000,
    help='point to plane loss weight'
)

parser.add_argument(
    '--chamfer_weight',
    type=float,
    default=300,
    help='chamfer loss weight'
)

parser.add_argument(
    '--shape_fitting_frames',
    type=int,
    default=10,
    help='Number of frames to use for shape fitting'
)

parser.add_argument(
    '--use_chamfer',
    type=bool,
    default=True,
    help='If true use chamfer loss instead of point to point and point to plane'
)

parser.add_argument(
    '--use_color',
    type=bool,
    default=True,
    help='If true use photometric loss'
)

def get_config(path_to_data_dir: str ='') -> argparse.Namespace:
    config = parser.parse_args()

    assert 0 < config.shape_params <= 300, "Shape params should be between 1 and 300"
    assert 0 < config.expression_params <= 100, "Shape params should be between 1 and 100"

    config.flame_model_path = path_to_data_dir + config.flame_model_path
    config.static_landmark_embedding_path = path_to_data_dir + config.static_landmark_embedding_path
    config.dynamic_landmark_embedding_path = path_to_data_dir + config.dynamic_landmark_embedding_path
    config.cam_data_dir = path_to_data_dir + config.cam_data_dir
    config.dlib_face_predictor_path = path_to_data_dir + config.dlib_face_predictor_path
    if config.device is None:
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return config
