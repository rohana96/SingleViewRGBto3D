"""
Usage:
    python -m src.q1_2 --num_frames 10 --image_size 256 --output_file images/q1/dolly_gif.gif
"""
import sys
sys.path.append("..")
import argparse
import imageio
import numpy as np
import pytorch3d
import torch
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from LearningPytorch3D.starter.utils import get_device, get_mesh_renderer


def dolly_zoom(
        cow_path="data/cow_on_plane.obj",
        image_size=256,
        num_frames=10,
        device=None,
):
    if device is None:
        device = get_device()

    mesh = pytorch3d.io.load_objs_as_meshes([cow_path])
    mesh = mesh.to(device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, -3.0]], device=device)

    fovs = torch.linspace(5, 120, num_frames)
    renders = []
    for fov in tqdm(fovs):
        distance = 6. / (2 * torch.tan(fov / 2 * np.pi / 180))
        T = [[0, 0, distance]]
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(fov=fov, T=T, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy()  # (N, H, W, 3)
        renders.append(rend)

    return renders


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow_on_plane.obj")
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--duration", type=float, default=3)
    parser.add_argument("--output_file", type=str, default="images/q1/dolly_gif.gif")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    images = dolly_zoom(
        image_size=args.image_size,
        num_frames=args.num_frames,
        cow_path=args.cow_path
    )
    fps = (args.num_frames / args.duration)
    imageio.mimsave(args.output_file, images, fps=fps)
