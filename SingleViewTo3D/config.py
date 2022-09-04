"""Config file"""

import torch

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
OUT_PATH_POINT_CLOUD = 'out_point_cloud/'
OUT_PATH_MESH = 'out_mesh/'
OUT_PATH_VOXEL = 'out_voxel/'
OUT_PATH_IMPLICIT = 'out_implicit/'


