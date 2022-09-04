from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d
from decoders import decoder_mesh, decoder_voxel, decoder_point_cloud
from encoders import encoder


class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = "cuda"
        self.encoder = encoder.Encoder(args)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.encoded_feat = None
        # define decoder
        if args.type == "vox":
            self.decoderVoxel = decoder_voxel.LinearDecoderVoxel()
        elif args.type == "point":
            self.n_point = args.n_points
            self.decoderPointCloud = decoder_point_cloud.DecoderPointCloud(n_points=self.n_point)
        elif args.type == "mesh":
            # try different mesh initializations
            if torch.cuda.is_available():
                mesh_pred = ico_sphere(4, 'cuda')
            else:
                mesh_pred = ico_sphere(4, 'cpu')

            color = [0.8, 0.8, 0.8]
            n_point = mesh_pred.verts_list()[0].shape[0]
            textures = torch.ones(size=(args.batch_size, n_point, 3))  # (1, N_v, 3)
            textures = textures * torch.tensor(color)
            if torch.cuda.is_available():
                textures = textures.cuda()  # (1, N_v, 3)
            self.mesh_pred = pytorch3d.structures.Meshes(
                mesh_pred.verts_list() * args.batch_size,
                mesh_pred.faces_list() * args.batch_size,
                textures=pytorch3d.renderer.TexturesVertex(textures)
            )

            self.decoderMesh = decoder_mesh.DecoderMesh(n_points=n_point)

    def forward(self, images, args):
        results = dict()
        total_loss = 0.0
        start_time = time.time()

        self.encoded_feat = self.encoder(images)

        # call decoder
        if args.type == "vox":
            voxels_pred = self.decoderVoxel(self.encoded_feat)
            return voxels_pred

        elif args.type == "point":
            pointclouds_pred = self.decoderPointCloud(self.encoded_feat)
            return pointclouds_pred

        elif args.type == "mesh":
            deform_vertices_pred = self.decoderMesh(self.encoded_feat)
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1, 3]))
            return mesh_pred
