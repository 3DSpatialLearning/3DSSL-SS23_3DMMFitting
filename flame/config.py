import argparse

parser = argparse.ArgumentParser(description='FLAME model')

parser.add_argument(
    '--flame_model_path',
    type=str,
    default='../data/flame_model/generic_model.pkl',
    help='flame model path'
)

parser.add_argument(
    '--static_landmark_embedding_path',
    type=str,
    default='../data/flame_model/flame_static_embedding.pkl',
    help='Static landmark embeddings path for FLAME'
)

parser.add_argument(
    '--dynamic_landmark_embedding_path',
    type=str,
    default='../data/flame_model/flame_dynamic_embedding.npy',
    help='Dynamic contour embedding path for FLAME'
)

parser.add_argument(
    '--tex_space_path',
    type=str,
    default='../data/flame_model/FLAME_texture.npz',
    help='Texture space path for FLAME'
)

# FLAME hyper-parameters
parser.add_argument(
    '--shape_params',
    type=int,
    default=100,
    help='the number of shape parameters'
)

parser.add_argument(
    '--expression_params',
    type=int,
    default=50,
    help='the number of expression parameters'
)

parser.add_argument(
    '--tex_params',
    type=int,
    default=50,
    help='the number of expression parameters'
)

parser.add_argument(
    '--pose_params',
    type=int,
    default=6,
    help='the number of pose parameters'
)

# Training hyper-parameters
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

parser.add_argument(
    '--learning_rate_first_frame',
    type=float,
    default=0.01,
    help='learning rate for first frame fitting'
)

parser.add_argument(
    '--num_opt_steps_first_frame',
    type=int,
    default=200,
    help='optimizing steps for first frame fitting'
)

parser.add_argument(
    '--learning_rate_subsequent_frames',
    type=float,
    default=0.001,
    help='learning rate for subsequent frames fitting'
)

parser.add_argument(
    '--num_opt_steps_subsequent_frames',
    type=int,
    default=50,
    help='optimizing steps for subsequent frames fitting'
)

parser.add_argument(
    '--landmark_weight',
    type=float,
    default=1e-2,
    help='landmark loss weight'
)

parser.add_argument(
    '--rgb_weight',
    type=float,
    default=1e-2,
    help='color loss weight'
)

parser.add_argument(
    '--point2point_weight',
    type=float,
    default=1e-3,
    help='point to point loss weight'
)

parser.add_argument(
    '--point2plane_weight',
    type=float,
    default=1e-3,
    help='point to plane loss weight'
)

parser.add_argument(
    'shape_weight',
    type=float,
    default=1e-8,
    help='shape regularization strength'
)

parser.add_argument(
    'exp_weight',
    type=float,
    default=1e-9,
    help='expression regularization strength'
)

parser.add_argument(
    'tex_weight',
    type=float,
    default=1e-8,
    help='texture regularization strength'
)


def get_config():
    config = parser.parse_args()
    return config
