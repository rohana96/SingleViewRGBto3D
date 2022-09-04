"""
Usage:
    python -m src.q4 --image_size 512 --output_path images/q4/camera_1.jpg images/q4/camera_2.jpg images/q4/camera_3.jpg
    images/q4/camera_4.jpg
"""
import sys
sys.path.append("..")
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pytorch3d
import torch
from pytorch3d.vis.plotly_vis import plot_scene

from LearningPytorch3D.starter.utils import get_device, get_mesh_renderer


def render_textured_cow(
        cow_path="data/cow.obj",
        image_size=256,
        R_relative=np.eye(3),
        T_relative=np.array([0, 0, 0]),
        device=None,
):
    if device is None:
        device = get_device()
    meshes = pytorch3d.io.load_objs_as_meshes([cow_path]).to(device)

    R_relative = torch.FloatTensor(R_relative)
    T_relative = torch.FloatTensor(T_relative)
    R = R_relative @ torch.tensor([
        [1.0, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    T = R_relative @ torch.tensor([0.0, 0, 3]) + T_relative

    # print("Final R=", R)
    # print("Final T=", T)

    renderer = get_mesh_renderer(image_size=256)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R.unsqueeze(0), T=T.unsqueeze(0), device=device,
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device, )
    rend = renderer(meshes, cameras=cameras, lights=lights)

    # fig = plot_scene({
    #     "some_name": {
    #         "meshes": meshes,
    #         "camera": cameras
    #     }
    # })
    # fig.show()

    return rend[0, ..., :3].cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument(
        "--output_path",
        type=str,
        nargs="+",
        default=[
            "images/q4/camera_transforms_1.jpg",
            "images/q4/camera_transforms_2.jpg",
            "images/q4/camera_transforms_3.jpg",
            "images/q4/camera_transforms_4.jpg",
        ])
    args = parser.parse_args()

    # # 4.1
    R1 = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ]).T
    T1 = np.array([0, 0, 0])

    R2 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]).T
    T2 = np.array([0, 0, 3])

    R3 = np.identity(3).T
    T3 = np.array([0.5, -0.5, 0])

    R4 = np.array([
        [0, 0, -1],
        [0, 1, 0],
        [1, 0, 0]
    ]).T
    T4 = np.array([-3, 0, 3])

    imgs = [
        render_textured_cow(
            cow_path=args.cow_path,
            image_size=args.image_size,
            R_relative=R,
            T_relative=T)
        for (R, T)
        in [(R1, T1), (R2, T2), (R3, T3), (R4, T4)]
    ]
    for path, img in zip(args.output_path, imgs):
        plt.imsave(path, img)
