"""
Usage:
    python -m src.q6 --render parametric --num_samples 3000 --image_size 256 --output_path images/q6/spiral_parametric.gif
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


def render_torus(image_size=256, num_samples=2000, device=None):
    """
    Renders a sphere using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    R = 1
    y = torch.linspace(-3, 3, num_samples)
    # Densely sample phi and theta on a grid
    f = 150
    x = y*R*torch.cos(f*y)
    z = y*R*torch.sin(f*y)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    torus_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    return get_point_cloud_images(point_cloud=torus_point_cloud, light_location=[0, 0.0, -4.0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", type=str, default="parametric")
    parser.add_argument("--output_path", type=str, default="images/q6/spiral_parametric.gif")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    images = render_torus()
    imageio.mimsave(args.output_path, images, fps=5)
