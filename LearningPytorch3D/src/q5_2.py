"""
Usage:
    python -m src.q5_2 --render parametric --num_samples 100 1000 --image_size 256 --output_path images/q5/
"""
import sys
sys.path.append("..")
import argparse

import imageio
import numpy as np
import pytorch3d
import torch

from LearningPytorch3D.src.utils import get_point_cloud_images
from LearningPytorch3D.starter.utils import get_device


def render_torus(image_size=256, num_samples=100, device=None):
    """
    Renders a torus using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    R = 0.75
    r = 0.25
    phi = torch.linspace(0, 2*np.pi, num_samples)
    theta = torch.linspace(0, 2*np.pi, num_samples)
    # Densely sample phi and theta on a grid
    phi, theta = torch.meshgrid(phi, theta)

    x = (R + r * torch.cos(phi)) * torch.cos(theta)
    y = (R + r * torch.cos(phi)) * torch.sin(theta)
    z = r * torch.sin(phi)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())
    torus_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    return get_point_cloud_images(point_cloud=torus_point_cloud, light_location=[0, 0.0, -4.0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", type=str, default="parametric")
    parser.add_argument("--output_path", type=str, default="images/q5/")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", nargs='+', type=int, default=[100, 1000])
    args = parser.parse_args()
    for num_sample in args.num_samples:
        file_path = "{0}torus_parametric_{1}.gif".format(args.output_path, num_sample)
        images = render_torus(num_samples=num_sample, image_size=args.image_size)
        imageio.mimsave(file_path, images, fps=5)
