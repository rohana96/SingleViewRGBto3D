"""Implements utility functions"""

import os
import numpy as np
import torch
import torch.utils.data
from matplotlib import pyplot as plt

import config


def load_data(file_path):
    """ Load points and occupancy values from file.

    :param file_path: (string) path to file
    :return: ground truth points and occupancies
    """

    print("....loading data")
    data_dict = np.load(file_path)
    points = data_dict['points']
    occupancies = data_dict['occupancies']

    # Unpack data format of occupancies
    occupancies = np.unpackbits(occupancies)[:points.shape[0]]
    occupancies = occupancies.astype(np.float32)

    # Align z-axis with top of object
    points = np.stack([points[:, 0], -points[:, 2], points[:, 1]], 1)

    return points, occupancies


def train_val_split(points, occupancies):
    """
    Split data into train and validation set

    :param points: (torch.Tensor or np.ndarray): 3D coordinates of the points
    :param occupancies: (torch.Tensor or np.ndarray): occupancy values for the points
    :return: training and validation sets
    """
    print("....splitting data intro train and val sets")

    n = len(points)
    indices = np.random.permutation(n)
    train_idx, val_idx = indices[:int(0.8 * n)], indices[int(0.8 * n):]

    train_points, val_points = points[train_idx], points[val_idx]
    train_occs, val_occs = occupancies[train_idx], occupancies[val_idx]

    train_set = torch.utils.data.TensorDataset(torch.from_numpy(train_points), torch.from_numpy(train_occs))
    val_set = torch.utils.data.TensorDataset(torch.from_numpy(val_points), torch.from_numpy(val_occs))

    return train_set, val_set


def save_checkpoint(model, optimizer, save_dir, epoch, representation):
    """
    Save model checkpoints

    :param representation:
    :param model: model
    :param optimizer: optimizer
    :param save_dir: directory to store model checkpoints
    :param epoch: poch number
    """

    filename = f'checkpoint_{representation}_{epoch}.pth'
    filepath = save_dir + filename
    print(".....saving checkpoint")
    checkpoint = {
        "step": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(checkpoint_dir, representation, model, optimizer=None, run_type="eval", idx=9999):
    """
    Load saved checkpoint

    :param run_type: "train" or "eval"
    :param optimizer: optimizer
    :param model: model
    :param checkpoint_dir: (str): path to saved checkpoint
    :param representation: "vox", "point", "mesh"
    :param idx: checkpoint index
    :return:
    """
    # f'{checkpoint_dir}{representation}/checkpoint_{representation}_{idx}.pth'
    checkpoint = torch.load(checkpoint_dir, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_iter = 0
    if run_type == "train":
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint['step']
    return start_iter


def visualize_occupancy(points, occupancies, id, save_dir="", n=50000):
    """ Visualize points and occupancy values.

    :param points: (n_points, 3) torch.Tensor 3D coordinates of the points
    :param occupancies: (n_points, ) torch.Tensor occupancy values for the points
    :param id: unique id for the visual
    :param save_dir: visualization save directory
    :param n: maximum number of points to visualize
    """
    occupancies = occupancies > 0.
    # if needed convert torch.tensor to np.ndarray
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if isinstance(occupancies, torch.Tensor):
        occupancies = occupancies.detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    n = min(len(points), n)

    # visualize a random subset of n points
    idcs = np.random.randint(0, len(points), n)
    points = points[idcs]
    occupancies = occupancies[idcs]

    # define colors
    red = np.array([1, 0, 0, 0.5]).reshape(1, 4).repeat(n, 0)  # plot occupied points in red with alpha=0.5
    blue = np.array([0, 0, 1, 0.01]).reshape(1, 4).repeat(n, 0)  # plot free points in blue with alpha=0.01
    occ = occupancies.reshape(n, 1).repeat(4, 1)  # reshape to RGBA format to determine color
    color = np.where(occ == 1, red, blue)  # occ=1 -> red, occ=0 -> blue

    # plot the points
    ax.scatter(*points.transpose(), color=color)

    # make it pretty
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(0, 32.0)
    ax.set_ylim(0, 32.0)
    ax.set_zlim(0, 32.0)
    plt.savefig(f'{save_dir}implicit_{id}.png')


def make_grid(xmin, xmax, resolution):
    """
    Create equidistant points on 3D grid (cube shaped)

    :param xmin: (float): minimum for x,y,z
    :param xmax: (float): number of hidden layers
    :param resolution: grid size
    :return: Tensor: (n, 3) tensor where n is r**3
    """
    grid_1d = torch.linspace(xmin, xmax, resolution)
    grid_3d = torch.stack(torch.meshgrid(grid_1d, grid_1d, grid_1d), -1)
    return grid_3d.flatten(0, 2)


def sample_points_from_voxel(voxel_grid, n_points=5000):
    """
    Sample n_points from a voxel grid

    :param voxel_grid: (n_batch, vox_size, vox_size, vox_size) dimensional grid
    :param n_points: number of points to sample
    :return: input batch of points and their ground truth occupancy values
    """

    n_batch = voxel_grid.shape[0]
    vox_size = voxel_grid.shape[-1]
    input_points = torch.rand(size=(n_batch, n_points, 3)) * vox_size
    occupancy_grid = torch.zeros(n_batch, n_points, 1)
    input_points_floor = torch.floor(input_points).type(torch.long)

    temp = [
        voxel_grid[i, 0, input_points_floor[i,:,0], input_points_floor[i,:,1], input_points_floor[i,:,2]]
        for i in range(n_batch)
    ]
    occupancy_grid[:, :, 0] = torch.stack(temp, dim = 0)
    # occupancy_grid[:, :, 0] = torch.tensor([voxel_grid[i, 0, input_points_floor[i, :, 0], input_points_floor[i, :, 1], input_points_floor[i, :, 2]] for i in range(n_batch)])
    if torch.cuda.is_available():
        return input_points.float().cuda(), occupancy_grid.float().cuda()
    return input_points.float(), occupancy_grid.float()


def sample_voxel_grid_points_for_eval(batch_size=1, vox_size=32):
    """
    :param batch_size:
    :param vox_size:
    :return:
    """
    x = torch.arange(0, vox_size)
    y = torch.arange(0, vox_size)
    z = torch.arange(0, vox_size)
    num_points = torch.cartesian_prod(x, y, z)
    # X, Y, Z = torch.meshgrid(x, y, z)
    # #torch.cartesian_pro
    # num_points = torch.stack([X.reshape(1, -1), Y.reshape(1, -1), Z.reshape(1, -1)], dim=2)
    input_points = num_points.repeat(batch_size, 1, 1).float()
    if torch.cuda.is_available():
        input_points = input_points.cuda()
    return input_points.float()

# if __name__ == "__main__":
#     points, occupancies = load_data(config.FILE_PATH)
#     visualize_occupancy(points, occupancies)
