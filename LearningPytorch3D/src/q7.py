"""
Usage:
    python -m src.q7 --image_size 256 --output_path images/q7/
"""
import sys

sys.path.append("..")
import argparse
import imageio
import numpy as np
import pytorch3d
import torch

from LearningPytorch3D.src.utils import get_point_cloud_images
from LearningPytorch3D.starter.utils import get_device, get_points_renderer, load_mesh


def render_point_cloud(
        cow_path="data/cow.obj",
        image_size=256,
        color=[0.7, 0.7, 1],
        background_color=(0, 0, 0),
        device=None,
        num_sample=1000
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )

    vertices, faces = load_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    face_areas = mesh.faces_areas_packed()
    face_prob = face_areas / torch.sum(face_areas)

    n_faces = face_prob.shape[0]
    sampled_faces = np.random.choice(n_faces, size=num_sample, p=face_prob.numpy())
    verts = torch.zeros((num_sample, 3))

    for i, face in enumerate(sampled_faces):
        vert = faces[0, face, :].tolist()
        p1, p2, p3 = vertices[0, vert, :]
        alpha = np.random.uniform(low=0.0, high=1.0)
        alpha2 = np.random.uniform(low=0.0, high=1.0)
        alpha1 = 1 - np.sqrt(alpha)
        p = alpha1 * p1 + (1 - alpha1) * alpha2 * p2 + (1 - alpha1) * (1 - alpha2) * p3
        verts[i, :] = p

    rgb = torch.empty(size=(num_sample, 3))
    rgb[:, :] = torch.Tensor(color)
    verts = torch.Tensor(verts).to(device).unsqueeze(0)
    rgb = torch.Tensor(rgb).to(device).unsqueeze(0)
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)

    return get_point_cloud_images(point_cloud, light_location=[[0, 0, -4]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--num_sample", type=int, nargs='+', default=[10, 100, 1000, 10000])
    parser.add_argument("--output_path", type=str, default="images/q7/")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    for num_sample in args.num_sample:
        file_path = "{0}cow_point_cloud_{1}.gif".format(args.output_path, num_sample)
        images = render_point_cloud(cow_path=args.cow_path, num_sample=num_sample, image_size=args.image_size)
        imageio.mimsave(file_path, images, fps=5)
