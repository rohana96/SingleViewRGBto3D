"""
Usage:
    python -m src.q5_1 --render point_cloud --image_size 256 --output_path images/q5/plants1.gif images/q5/plants2.gif images/q5/plants3.gif
"""
import sys
sys.path.append("..")
import argparse
import pickle

import imageio
import pytorch3d
import torch

from LearningPytorch3D.src.utils import get_point_cloud_images
from LearningPytorch3D.starter.utils import get_device, unproject_depth_image


def load_rgbd_data(path="data/rgbd_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def render_plant(
        data_path="data/rgbd_data.pkl",
        image_size=256,
        background_color=(1, 1, 1),
        device=None,
):
    """
    Returns 2D projects of 2 point clouds and their union
    """

    if device is None:
        device = get_device()

    data_dict = load_rgbd_data(path=data_path)

    rgb1 = torch.Tensor(data_dict["rgb1"])
    rgb2 = torch.Tensor(data_dict["rgb2"])

    mask1 = torch.Tensor(data_dict["mask1"])
    mask2 = torch.Tensor(data_dict["mask2"])

    depth1 = torch.Tensor(data_dict["depth1"])
    depth2 = torch.Tensor(data_dict["depth2"])

    cameras1 = data_dict["cameras1"]
    cameras2 = data_dict["cameras2"]

    point_cloud1, rgba1 = unproject_depth_image(image=rgb1, mask=mask1, depth=depth1, camera=cameras1)
    point_cloud2, rgba2 = unproject_depth_image(image=rgb2, mask=mask2, depth=depth2, camera=cameras2)

    # print(type(point_cloud1), point_cloud1[0], point_cloud1.shape)
    verts1 = torch.Tensor(point_cloud1[::10]).to(device).unsqueeze(0)
    rgb1 = torch.Tensor(rgba1[::10]).to(device).unsqueeze(0)

    verts2 = torch.Tensor(point_cloud2[::10]).to(device).unsqueeze(0)
    rgb2 = torch.Tensor(rgba2[::10]).to(device).unsqueeze(0)

    verts3 = torch.cat((verts1, verts2), dim=1)
    rgb3 = torch.cat((rgb1, rgb2), dim=1)

    point_cloud1 = pytorch3d.structures.Pointclouds(points=verts1, features=rgb1)
    point_cloud2 = pytorch3d.structures.Pointclouds(points=verts2, features=rgb2)
    point_cloud3 = pytorch3d.structures.Pointclouds(points=verts3, features=rgb3)

    R_relative = torch.tensor([[1.0, 0, 0], [0, -1, 0], [0, 0, 1]]).type(torch.FloatTensor)
    T_relative = torch.tensor([0.0, 0.0, 3])


    my_images1 = get_point_cloud_images(point_cloud1, R_relative=R_relative, T_relative=T_relative)
    my_images2 = get_point_cloud_images(point_cloud2, R_relative=R_relative, T_relative=T_relative)
    my_images3 = get_point_cloud_images(point_cloud3, R_relative=R_relative, T_relative=T_relative)

    return my_images1, my_images2, my_images3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", type=str, default="implicit", )
    parser.add_argument(
        "--output_path",
        type=str,
        nargs="+",
        default=[
            "images/q5/plants1_gif.gif",
            "images/q5/plants2_gif.gif",
            "images/q5/plants3_gif.gif"
        ])

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    images1, images2, images3 = render_plant()
    imageio.mimsave(args.output_path[0], images1, fps=5)
    imageio.mimsave(args.output_path[1], images2, fps=5)
    imageio.mimsave(args.output_path[2], images3, fps=5)
