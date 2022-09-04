"""
Usage:
    python -m src.q1_1 --image_size 256 --output_path images/q1/cow_render.gif
"""
import imageio
import argparse
import sys
sys.path.append("..")
from LearningPytorch3D.src.utils import get_mesh_images, get_mesh
from LearningPytorch3D.starter.utils import get_device


def render_gif(
        cow_path="/data/cow.obj", image_size=256, color=[0.7, 0.7, 1], device=None,
):
    if device is None:
        device = get_device()
    mesh = get_mesh(cow_path=cow_path, image_size=256, color=color)
    return get_mesh_images(mesh=mesh, light_location=[[0, 0, 4]], image_size=image_size, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_path", type=str, default="images/q1/cow_render.gif")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    images = render_gif(cow_path=args.cow_path, image_size=args.image_size)
    imageio.mimsave(args.output_path, images, fps=5)
