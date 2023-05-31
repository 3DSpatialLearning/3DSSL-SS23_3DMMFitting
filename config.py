import argparse

parser = argparse.ArgumentParser(description='FLAME fitting config')


#################### FLAME args ####################

parser.add_argument(
    '--flame_model_path',
    type=str,
    default='data/flame_model/generic_model.pkl',
    help='flame model path'
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
    '--pose_params',
    type=int,
    default=6,
    help='the number of pose parameters'
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

####################### Other args #######################

parser.add_argument(
    '--steps',
    type=int,
    default=200,
    help='Per frame optimizing steps'
)

parser.add_argument(
    '--lr',
    type=float,
    default=1e-2,
    help='Learning rate'
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
    '--landmark_weight',
    type=float,
    default=1e-2,
    help='Landmark term weight'
)

parser.add_argument(
    '--shape_regularization_weight',
    type=float,
    default=1e-8,
    help='Shape regularization weight'
)

parser.add_argument(
    '--exp_regularization_weight',
    type=float,
    default=1e-9,
    help='Expression regularization weight'
)

def get_config(path_to_data_dir: str ='') -> argparse.Namespace:
    config = parser.parse_args()
    config.flame_model_path = path_to_data_dir + config.flame_model_path
    config.static_landmark_embedding_path = path_to_data_dir + config.static_landmark_embedding_path
    config.dynamic_landmark_embedding_path = path_to_data_dir + config.dynamic_landmark_embedding_path
    assert config.shape_params > 0 and config.shape_params <= 300, "Shape params should be between 1 and 300"
    assert config.expression_params > 0 and config.expression_params <= 100, "Shape params should be between 1 and 100"
    return config
