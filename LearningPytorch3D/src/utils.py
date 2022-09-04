import sys

sys.path.append("..")
import imageio
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pytorch3d
import torch
from tqdm import tqdm
from LearningPytorch3D.starter.utils import get_device, get_mesh_renderer, get_points_renderer, load_mesh


def get_mesh(
        cow_path="data/cow.obj",
        image_size=256,
        color=None,
        device=None,
):

    if color is None:
        color = [0.7, 0.7, 1]

    if device is None:
        device = get_device()

    renderer = get_mesh_renderer(image_size=image_size)
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
    return mesh


def get_mesh_images(
        mesh,
        light_location=None,
        num_images=15,
        image_size=256,
        device=None,
        T_relative=torch.tensor([0.0, 0.0, 0.0]),
        R_relative=torch.eye(3)
):
    """
    Renders a set 2D projections of a mesh
    """

    if mesh.textures is None:
        color = [0.7, 0.7, 1]
        textures = torch.FloatTensor(color).unsqueeze(0).repeat(
            mesh.verts_list()[0].shape[0], 1
        )
        textures = textures.unsqueeze(0)
        if torch.cuda.is_available():
            textures = textures.cuda()
        mesh.textures = pytorch3d.renderer.TexturesVertex(textures)

    if light_location is None:
        light_location = [[0, 0, -4]]
    if device is None:
        device = get_device()

    R_relative = R_relative.to(device)
    T_relative = T_relative.to(device)
    mesh = mesh.to(device)
    my_images = []
    renderer = get_mesh_renderer(image_size=image_size)
    lights = pytorch3d.renderer.PointLights(location=light_location, device=device)

    azim = np.linspace(start=0, stop=360, num=num_images)
    elev = np.linspace(start=0, stop=360, num=num_images)
    xv, yv = np.concatenate((azim, np.zeros_like(azim))), np.concatenate((np.zeros_like(elev), elev))

    R, T = pytorch3d.renderer.cameras.look_at_view_transform(
        dist=4.0, elev=yv, azim=xv, degrees=True, eye=None, at=((0, 0, 0),),
        up=((0, 1, 0),), device=device)

    for R, T in tqdm(zip(R, T)):
        T = T_relative + T
        R = R_relative @ R
        T = T.unsqueeze(0)
        R = R.unsqueeze(0)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )

        rend = renderer(mesh, cameras=cameras, lights=lights, device=device)
        rend = rend.detach().cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        my_images.append(rend)  # List of images [(H, W, 3)]
    return my_images


def get_point_cloud_images(
        point_cloud,
        image_size=256,
        light_location=None,
        background_color=(0, 0, 0),
        num_images=15,
        device=None,
        T_relative=torch.tensor([0.0, 0.0, 0.0]),
        R_relative=torch.eye(3)
):
    """
    Renders a set 2D projections of a point cloud
    """
    if light_location is None:
        light_location = [[0, 0, -4]]
    if device is None:
        device = get_device()

    R_relative = R_relative.to(device)
    T_relative = T_relative.to(device)
    my_images = []
    renderer = get_points_renderer(image_size=image_size, background_color=background_color)
    lights = pytorch3d.renderer.PointLights(location=light_location, device=device, )

    azim = np.linspace(start=0, stop=360, num=num_images)
    elev = np.linspace(start=0, stop=360, num=num_images)
    xv, yv = np.concatenate((azim, np.zeros_like(azim))), np.concatenate((np.zeros_like(elev), elev))

    R, T = pytorch3d.renderer.cameras.look_at_view_transform(
        dist=4.0, elev=yv, azim=xv, degrees=True, eye=None, at=((0, 0, 0),),
        up=((0, 1, 0),), device=device)

    for R, T in tqdm(zip(R, T)):
        T = T_relative + T
        R = R_relative @ R
        T = T.unsqueeze(0)
        R = R.unsqueeze(0)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )
        rend = renderer(point_cloud, cameras=cameras, lights=lights, device=device)
        rend = rend.detach().cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        my_images.append(rend)  # List of images [(H, W, 3)]
    return my_images
