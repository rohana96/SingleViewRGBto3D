import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from visualizer.visualize_mesh import VisualizeMesh
from visualizer.visualize_point_cloud import VisualizePointCloud
from visualizer.visualize_voxel_grid import VisualizeVoxel
from model import SingleViewto3D
from r2n2_custom import R2N2
from pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
from utils import load_checkpoint

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
    parser.add_argument('--vis_freq', default=1, type=str)
    parser.add_argument('--batch_size', default=2, type=str)
    parser.add_argument('--num_workers', default=1, type=str)
    parser.add_argument('--type', default='point', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)
    parser.add_argument('--load_checkpoint', default=True, type=bool)
    parser.add_argument('--checkpoint_dir', default='checkpoints/', type=str)
    parser.add_argument('--save_dir', default='out_visual_latent/', type=str)
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
    if args.load_checkpoint:
        load_checkpoint(checkpoint_dir=args.checkpoint_dir, representation=args.type, model=model)
        print(f"Succesfully loaded iter {start_iter}")

    visualizer = VISUALIZER_DICT[args.type]

    print("Starting evaluating !")
    max_iter = min(len(eval_loader), 10)

    for step in range(start_iter, max_iter):

        feed_dict = next(eval_loader)
        images_gt, mesh_gt = preprocess(feed_dict)
        predictions = model(images_gt, args)

        feat1, feat2 = model.encoded_feat
        image1, image2 = images_gt.detach().cpu().numpy()
        feat1, feat2 = feat1.unsqueeze(0), feat2.unsqueeze(0)

        filepath = f'{args.save_dir}{args.run_type}/{args.type}/{step}/'
        os.makedirs(filepath, exist_ok=True)

        plt.imsave(f'{filepath}image_1.png', image1)
        plt.imsave(f'{filepath}image_2.png', image2)

        if args.type == "mesh":
            decoder = model.decoderMesh
        elif args.type == "point":
            decoder = model.decoderPointCloud
        elif args.type == "vox":
            decoder = model.decoderVoxel

        for i, alpha in enumerate(np.linspace(start=0, stop=1, endpoint=True, num=15)):

            feat = alpha * feat1 + (1 - alpha) * feat2

            pred_feat_interp = decoder(feat)
            pred_direct_interp = alpha * decoder(feat1) + \
                                 (1 - alpha) * decoder(feat2)

            # if args.type == "mesh":
            #     # predicted mesh with feature interpolation
            #     pred_feat_interp = model.mesh_pred.offset_verts(pred_feat_interp.reshape((-1, 3)))
            #     # predicted mesh with render interpolation
            #     pred_direct_interp = model.mesh_pred.offset_verts(pred_direct_interp.reshape((-1, 3)))

            visualizer(pred_feat_interp, f'{i}_feat_interp', save_dir=filepath)
            visualizer(pred_direct_interp, f'{i}_direct_interp', save_dir=filepath)

    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    evaluate_model(args)
