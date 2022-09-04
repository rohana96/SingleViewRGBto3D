import argparse
import os
import time
import torch
from visualizer.visualize_mesh import VisualizeMesh
from visualizer.visualize_point_cloud import VisualizePointCloud
from visualizer.visualize_voxel_grid import VisualizeVoxel
from model import SingleViewto3D
from r2n2_custom import R2N2
from pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
from pytorch3d.ops import sample_points_from_meshes
from losses import mesh_loss, point_cloud_loss, voxel_loss
from tqdm import tqdm
from utils import load_checkpoint, save_checkpoint

import warnings

warnings.filterwarnings("ignore")

VISUALIZER_DICT = {
    "point": VisualizePointCloud(),
    "mesh": VisualizeMesh(),
    "vox": VisualizeVoxel()
}


def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    # Model parameters
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--lr', default=4e-4, type=str)
    parser.add_argument('--max_iter', default=10000, type=int)
    parser.add_argument('--log_freq', default=1000, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--type', default='mesh', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.3, type=float)
    parser.add_argument('--save_freq', default=500, type=int)
    parser.add_argument('--save_dir', default='out_visual/', type=str)
    parser.add_argument('--run_type', default='train', type=str)
    parser.add_argument('--load_checkpoint', default=False, type=bool)
    parser.add_argument('--checkpoint_dir', default='checkpoints/', type=str)
    return parser


def preprocess(feed_dict, args):
    """

    :param feed_dict:
    :param args:
    :return:
    """
    for k in ['images', 'voxels', 'mesh']:
        if torch.cuda.is_available():
            feed_dict[k] = feed_dict[k].cuda()
        else:
            feed_dict[k] = feed_dict[k]
    images = feed_dict['images'].squeeze(1)
    if args.type == "vox":
        voxels = feed_dict['voxels'].float()
        ground_truth_3d = voxels
    elif args.type == "point":
        mesh = feed_dict['mesh']
        pointclouds_tgt = sample_points_from_meshes(mesh, args.n_points)
        ground_truth_3d = pointclouds_tgt
    elif args.type == "mesh":
        ground_truth_3d = feed_dict["mesh"]

    if torch.cuda.is_available():
        images, ground_truth_3d = images.cuda(), ground_truth_3d.cuda()
    return images, ground_truth_3d


def calculate_loss(predictions, ground_truth, args):
    """

    :param predictions:
    :param ground_truth:
    :param args:
    :return:
    """
    loss = 0
    if args.type == 'vox':
        loss = voxel_loss.VoxelLoss()(predictions, ground_truth)
    elif args.type == 'point':
        loss = point_cloud_loss.PointCloudLoss()(predictions, ground_truth)
    elif args.type == 'mesh':
        sample_trg = sample_points_from_meshes(ground_truth, args.n_points)
        sample_pred = sample_points_from_meshes(predictions, args.n_points)

        loss_reg = point_cloud_loss.PointCloudLoss()(sample_pred, sample_trg)
        loss_smooth = mesh_loss.MeshLoss()(predictions)
        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth
    return loss


def train_model(args):
    """

    :param args:
    """
    r2n2_dataset = R2N2("train", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True)

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True)
    train_loader = iter(loader)

    model = SingleViewto3D(args)
    if torch.cuda.is_available():
        model.cuda()
    model.train()

    # ============ preparing optimizer ... ============
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # to use with ViTs
    start_iter = 0
    start_time = time.time()

    visualizer = VISUALIZER_DICT[args.type]

    if args.load_checkpoint:
        checkpoint_dir = f'{args.checkpoint_dir}{args.type}/checkpoint_{args.type}_9999.pth'
        # checkpoint_dir = f'{args.checkpoint_dir}{args.type}_sm{args.w_smooth}/checkpoint_{args.type}_9999.pth'
        start_iter = load_checkpoint(checkpoint_dir=checkpoint_dir, representation=args.type, model=model, optimizer=optimizer, run_type="train")
        print(f"Successfully loaded iter {start_iter}")

    print("Starting training !")
    for step in tqdm(range(start_iter, args.max_iter)):
        iter_start_time = time.time()

        if step % len(train_loader) == 0:  # restart after one epoch
            train_loader = iter(loader)

        read_start_time = time.time()

        feed_dict = next(train_loader)

        images_gt, ground_truth_3d = preprocess(feed_dict, args)
        read_time = time.time() - read_start_time

        prediction_3d = model(images_gt, args)
        loss = calculate_loss(prediction_3d, ground_truth_3d, args)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        filepath = f'{args.checkpoint_dir}{args.type}_sm{args.w_smooth}/'
        os.makedirs(filepath, exist_ok=True)

        if ((step + 1) % args.save_freq) == 0:
            save_checkpoint(model=model, optimizer=optimizer, save_dir=filepath, epoch=step, representation=args.type)

            vis_save_path = f'{args.save_dir}{args.run_type}/{args.type}_sm{args.w_smooth}/'
            os.makedirs(vis_save_path, exist_ok=True)

            if args.type == "vox":
                prediction_3d = prediction_3d.permute(0, 1, 4, 3, 2).sigmoid()
            visualizer(prediction_3d[0], id=int(step / args.save_freq), save_dir=vis_save_path)

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f" % (step, args.max_iter, total_time, read_time, iter_time, loss_vis))

    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
