import argparse
import os
import time
import torch
import matplotlib.pyplot as plt
from visualizer.visualize_mesh import VisualizeMesh
from visualizer.visualize_point_cloud import VisualizePointCloud
from visualizer.visualize_voxel_grid import VisualizeVoxel
from model import SingleViewto3D
from r2n2_custom import R2N2
from pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import knn_points
from utils import load_checkpoint
import mcubes
import utils_vox

import warnings

warnings.filterwarnings("ignore")

VISUALIZER_DICT = {
    "point": VisualizePointCloud(),
    "mesh": VisualizeMesh(),
    "vox": VisualizeVoxel()
}


def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--max_iter', default=10000, type=str)
    parser.add_argument('--vis_freq', default= 25, type=str)
    parser.add_argument('--batch_size', default=1, type=str)
    parser.add_argument('--num_workers', default=4, type=str)
    parser.add_argument('--type', default='mesh', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.3, type=float)
    parser.add_argument('--load_checkpoint', default=True, type=bool)
    parser.add_argument('--checkpoint_dir', default='checkpoints/', type=str)
    parser.add_argument('--save_dir', default='out_visual/', type=str)
    parser.add_argument('--run_type', default='eval', type=str)

    return parser


def preprocess(feed_dict):
    for k in ['images']:
        if torch.cuda.is_available():
            feed_dict[k] = feed_dict[k].cuda()
        else:
            feed_dict[k] = feed_dict[k].cpu()

    images = feed_dict['images'].squeeze(1)
    mesh = feed_dict['mesh']

    return images, mesh


def compute_sampling_metrics(pred_points, gt_points, thresholds=[0.01, 0.02, 0.03, 0.04, 0.05], eps=1e-8):
    metrics = {}
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)

    # Compute precision, recall, and F1 based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics["Precision@%f" % t] = precision
        metrics["Recall@%f" % t] = recall
        metrics["F1@%f" % t] = f1

    # Move all metrics to CPU
    metrics = {k: v.cpu() for k, v in metrics.items()}
    return metrics


def evaluate(predictions, mesh_gt, args):
    pred_points = None
    if args.type == "vox":
        voxels_src = predictions
        H, W, D = voxels_src.shape[2:]
        vertices_src, faces_src = mcubes.marching_cubes(voxels_src.detach().cpu().squeeze().numpy(), isovalue=0.5)

        vertices_src = torch.tensor(vertices_src).float()
        faces_src = torch.tensor(faces_src.astype(int))
        mesh_src = pytorch3d.structures.Meshes([vertices_src], [faces_src])
        pred_points = sample_points_from_meshes(mesh_src, args.n_points)
        pred_points = utils_vox.Mem2Ref(pred_points, H, W, D)
    elif args.type == "point":
        pred_points = predictions.cpu()
    elif args.type == "mesh":
        pred_points = sample_points_from_meshes(predictions, args.n_points).cpu()

    gt_points = sample_points_from_meshes(mesh_gt, args.n_points)

    metrics = compute_sampling_metrics(pred_points, gt_points)
    return metrics


def evaluate_model(args):
    r2n2_dataset = R2N2("test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True)

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True)
    eval_loader = iter(loader)

    model = SingleViewto3D(args)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    start_iter = 0
    start_time = time.time()

    avg_f1_score = []

    if args.load_checkpoint:
        checkpoint_dir = f'{args.checkpoint_dir}{args.type}/checkpoint_{args.type}_9999.pth'
        # checkpoint_dir = f'{args.checkpoint_dir}{args.type}_sm{args.w_smooth}/checkpoint_{args.type}_9999.pth'
        load_checkpoint(checkpoint_dir=checkpoint_dir, representation=args.type, model=model)
        print(f"Succesfully loaded iter {start_iter}")

    visualizer = VISUALIZER_DICT[args.type]
    gt_visualizer = VISUALIZER_DICT["mesh"]

    print("Starting evaluating !")
    max_iter = len(eval_loader)

    for step in range(start_iter, max_iter):
        iter_start_time = time.time()

        read_start_time = time.time()

        feed_dict = next(eval_loader)

        images_gt, mesh_gt = preprocess(feed_dict)

        read_time = time.time() - read_start_time

        predictions = model(images_gt, args)

        if args.type == "vox":
            predictions = predictions.permute(0, 1, 4, 3, 2).sigmoid()
        metrics = evaluate(predictions, mesh_gt, args)

        filepath = f'{args.save_dir}{args.run_type}/{args.type}_sm{args.w_smooth}/'
        os.makedirs(filepath, exist_ok=True)

        if (step % args.vis_freq) == 0:
            visualizer(predictions, int(step / args.vis_freq), save_dir=filepath)
            gt_visualizer(mesh_gt, str(step / args.vis_freq) + "_gt", save_dir=filepath)
            plt.imsave(filepath + str(step / args.vis_freq) + "_image.png", images_gt[0].detach().cpu().numpy())

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        f1_05 = metrics['F1@0.050000']

        avg_f1_score.append(f1_05)

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); F1@0.05: %.3f; Avg F1@0.05: %.3f" % (
            step, max_iter, total_time, read_time, iter_time, f1_05, torch.tensor(avg_f1_score).mean()))

    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    evaluate_model(args)
