import argparse
import os
import time

from visualizer.visualize_mesh import VisualizeMesh
from visualizer.visualize_point_cloud import VisualizePointCloud
from visualizer.visualize_voxel_grid import VisualizeVoxel
from losses import mesh_loss, point_cloud_loss, voxel_loss
from pytorch3d.utils import ico_sphere
from r2n2_custom import R2N2
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
import dataset_location
import torch


def get_args_parser():
    """

    :return:
    """
    parser = argparse.ArgumentParser('Model Fit', add_help=False)
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--max_iter', default=100000, type=int)
    parser.add_argument('--log_freq', default=1000, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)
    parser.add_argument('--save_freq', default=1, type=int)
    parser.add_argument('--save_dir', default='out_visual_fit/', type=str)
    parser.add_argument('--run_type', default='fit', type=str)

    return parser


def fit_mesh(mesh_src, mesh_tgt, args):
    """
    :param mesh_src:
    :param mesh_tgt:
    :param args:
    """
    start_iter = 0
    start_time = time.time()

    if torch.cuda.is_available():
        deform_vertices_src = torch.zeros(mesh_src.verts_packed().shape, requires_grad=True, device='cuda')
    else:
        deform_vertices_src = torch.zeros(mesh_src.verts_packed().shape, requires_grad=True, device='cpu')
    optimizer = torch.optim.Adam([deform_vertices_src], lr=args.lr)
    criterion = mesh_loss.MeshLoss(reg_loss="laplacian")
    regularizer = point_cloud_loss.PointCloudLoss(method="chamfer")
    print("Starting training !")
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        new_mesh_src = mesh_src.offset_verts(deform_vertices_src)

        sample_trg = sample_points_from_meshes(mesh_tgt, args.n_points)
        sample_src = sample_points_from_meshes(new_mesh_src, args.n_points)

        loss_smooth = criterion(new_mesh_src, method="uniform")
        loss_reg = regularizer(sample_src, sample_trg)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time, iter_time, loss_vis))

    mesh_src.offset_verts_(deform_vertices_src)

    print('Done!')


def fit_pointcloud(pointclouds_src, pointclouds_tgt, args):
    start_iter = 0
    start_time = time.time()
    optimizer = torch.optim.Adam([pointclouds_src], lr=args.lr)
    criterion = point_cloud_loss.PointCloudLoss(method="chamfer")

    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = criterion(pointclouds_src, pointclouds_tgt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time, iter_time, loss_vis))

    print('Done!')


def fit_voxel(voxels_src, voxels_tgt, args):
    start_iter = 0
    start_time = time.time()
    optimizer = torch.optim.Adam([voxels_src], lr=args.lr)
    criterion = voxel_loss.VoxelLoss(method="BCEWithLogits")

    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = criterion(voxels_src, voxels_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time, iter_time, loss_vis))

    print('Done!')


def train_model(args):
    r2n2_dataset = R2N2("train", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True)

    VISUALIZER_DICT = {
        "point": VisualizePointCloud(),
        "mesh": VisualizeMesh(),
        "vox": VisualizeVoxel()
    }

    visualizer = VISUALIZER_DICT[args.type]
    feed = r2n2_dataset[0]

    feed_cuda = {}
    for k in feed:
        if torch.is_tensor(feed[k]):
            if torch.cuda.is_available():
                feed_cuda[k] = feed[k].cuda().float()
            else:
                feed_cuda[k] = feed[k].float()

    if args.type == "vox":
        # initialization
        if torch.cuda.is_available():
            voxels_src = torch.rand(feed_cuda['voxels'].shape, requires_grad=True, device='cuda')
        else:
            voxels_src = torch.rand(feed_cuda['voxels'].shape, requires_grad=True, device='cpu')
        voxel_coords = feed_cuda['voxel_coords'].unsqueeze(0)
        voxels_tgt = feed_cuda['voxels']

        # fitting
        vis_save_path = f'{args.save_dir}{args.run_type}/{args.type}/'
        os.makedirs(vis_save_path, exist_ok=True)
        fit_voxel(voxels_src, voxels_tgt, args)
        visualizer(voxels_src, id='pred', save_dir=vis_save_path)
        visualizer(voxels_tgt, id='gt', save_dir=vis_save_path)
        # visualize


    elif args.type == "point":
        # initialization
        if torch.cuda.is_available():
            pointclouds_src = torch.randn([1, args.n_points, 3], requires_grad=True, device='cuda')
        else:
            pointclouds_src = torch.randn([1, args.n_points, 3], requires_grad=True, device='cpu')
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])
        pointclouds_tgt = sample_points_from_meshes(mesh_tgt, args.n_points)

        # fitting
        fit_pointcloud(pointclouds_src, pointclouds_tgt, args)
        vis_save_path = f'{args.save_dir}{args.run_type}/{args.type}/'
        os.makedirs(vis_save_path, exist_ok=True)
        visualizer(pointclouds_src, id='pred', save_dir=vis_save_path)
        visualizer(pointclouds_tgt, id='gt', save_dir=vis_save_path)
        # visualize pt_cloud, gt

    elif args.type == "mesh":
        # initialization
        # try different ways of initializing the source mesh
        if torch.cuda.is_available():
            mesh_src = ico_sphere(4, 'cuda')
        else:
            mesh_src = ico_sphere(4, 'cpu')

        
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])

        # fitting
        fit_mesh(mesh_src, mesh_tgt, args)

        vis_save_path = f'{args.save_dir}{args.run_type}/{args.type}/'
        os.makedirs(vis_save_path, exist_ok=True)
        visualizer(mesh_src, id='pred', save_dir=vis_save_path)
        visualizer(mesh_tgt, id='gt', save_dir=vis_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model Fit', parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
