import torch
import einops

from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from einops.layers.torch import Rearrange

from emprot.models.backbone_distance_embedding import BackboneDistanceEmbedding

from emprot.utils.torch_utils import get_batches_to_idx
from emprot.utils.affine_utils import (
    sample_centered_cube_rot_matrix, 
    sample_centered_rectangle_along_vector,
)


class SpatialAvg(nn.Module):
    def forward(self, x):
        return x.mean(dim=[-3, -2, -1])

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        groups=1,
        activation_class=nn.ReLU,
        conv_class=nn.Conv3d,
        affine=False,
        checkpoint=True,
        **kwargs,
    ):
        super().__init__()
        self.activation_fn = activation_class()
        self.conv1 = conv_class(
            in_planes, planes, kernel_size=1, bias=False, groups=groups
        )
        self.bn1 = nn.InstanceNorm3d(planes, affine=affine)
        self.conv2 = conv_class(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups,
        )
        self.bn2 = nn.InstanceNorm3d(planes, affine=affine)
        self.conv3 = conv_class(
            planes, self.expansion * planes, kernel_size=1, bias=False, groups=groups
        )
        self.bn3 = nn.InstanceNorm3d(self.expansion * planes, affine=affine)

        self.shortcut_conv = nn.Identity()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut_conv = nn.Conv3d(
                in_planes,
                self.expansion * planes,
                kernel_size=1,
                stride=stride,
                bias=False,
                groups=groups,
            )

        self.forward = self.forward_checkpoint if checkpoint else self.forward_normal

    def forward_normal(self, x):
        out = self.activation_fn(self.bn1(self.conv1(x)))
        out = self.activation_fn(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut_conv(x)
        out = self.activation_fn(out)
        return out

    def forward_checkpoint(self, x):
        return torch_checkpoint(self.forward_normal, x, use_reentrant=False, preserve_rng_state=False)


class ConvBlock(nn.Module):
    def __init__(
        self, 
        in_channels:int, 
        out_channels:int,
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu1 = nn.ReLU()

    def forward(self,x):
        y = self.norm1(self.conv1(x))
        y = self.relu1(y)
        return y

class CryoInit(nn.Module):
    """
    Init density information to node and edge representation
    """
    def __init__(
        self,
        d_node: int = 256,
        d_edge: int = 128, 
        d_cryo_emb: int = 256, 
        cube_size : int = 23,
        rectangle_length : int = 17,
        k: int = 32, 
        activation_class:nn.Module = nn.ReLU,
        **kwargs,
    ):
        super().__init__()

        assert d_cryo_emb % 4 == 0

        self.k = k

        self.d_node = d_node
        self.d_edge = d_edge

        self.c_length = cube_size
        self.r_length = rectangle_length
        self.d_cryo_emb = d_cryo_emb

        self.activation_class = activation_class

        self.conv_cube = nn.Sequential(
            ConvBlock(1, self.d_cryo_emb),
            Bottleneck(self.d_cryo_emb, self.d_cryo_emb // 4, stride=2, affine=True),
            Bottleneck(
                self.d_cryo_emb, self.d_cryo_emb // 4, stride=2, affine=True
            ),
            Bottleneck(
                self.d_cryo_emb, self.d_cryo_emb // 4, stride=2, affine=True
            ),
            Bottleneck(self.d_cryo_emb, self.d_cryo_emb // 4, stride=2, affine=True),
            nn.Conv3d(self.d_cryo_emb, self.d_cryo_emb, kernel_size=2),
            SpatialAvg(),
            nn.Linear(self.d_cryo_emb, self.d_node, bias=False),
        )

        self.conv_rectangle_pre = nn.Conv3d(
            in_channels=1,
            out_channels=self.d_cryo_emb // 2,
            kernel_size=(1, 3, 3),
            bias=False,
        )

        # self.conv_rectangle_mid = rearrange
        # rearrange is dynamically used

        self.conv_rectangle_post = nn.Sequential(
            nn.LayerNorm(self.d_cryo_emb // 2 * self.r_length),
            activation_class(),
            nn.Linear(
                self.d_cryo_emb // 2 * self.r_length, self.d_edge, bias=False
            )
        )

        self.backbone_distance_emb = BackboneDistanceEmbedding()

    def forward(
        self,
        affines,
        cryo_grids=None,
        cryo_global_origins=None,
        cryo_voxel_sizes=None,
        batch=None,
        **kwargs,
    ):
        assert cryo_grids is not None

        batch_to_idx = (
            get_batches_to_idx(batch)
            if batch is not None
            else [torch.arange(0, len(affines), dtype=int, device=affines.device)]
        )

        max_l = max([len(idx) for idx in batch_to_idx])

        batch_affines = []
        #########
        # affines
        #########
        for idx in batch_to_idx:
            n = len(idx)
            new_affines = torch.cat([
                torch.eye(3)[None, ...].repeat(max_l, 1, 1).to(affines.device), # max_l, 3, 3
                torch.zeros(max_l, 3)[..., None].to(affines.device), # max_l, 3, 1
            ], dim=-1) # max_l, 3, 4

            new_affines[0 : n] = affines[idx]
            batch_affines.append(new_affines)
        batch_affines = torch.stack(batch_affines, dim=0)



        batch_node_repr = []
        batch_node_mask = []
        ###############
        # node features
        ###############
        with torch.no_grad():
            positions = affines[..., :3, -1]

            batch_cryo_grids = [
                cg.expand(len(b), -1, -1, -1, -1)
                for (cg, b) in zip(cryo_grids, batch_to_idx)
            ]
            cryo_points = [
                (positions[b].reshape(-1, 3) - go) / vz
                for (b, go, vz) in zip(
                    batch_to_idx, cryo_global_origins, cryo_voxel_sizes
                )
            ]

            cryo_points_rot_matrices = [
                affines[b][..., :3, :3].reshape(-1, 3, 3) for b in batch_to_idx
            ]

            cryo_points_cube = sample_centered_cube_rot_matrix(
                batch_cryo_grids,
                cryo_points_rot_matrices,
                cryo_points,
                cube_side=self.c_length,
            ) # n 1 c_len c_len c_len

        cryo_points_cube = self.conv_cube(cryo_points_cube.requires_grad_()) # n d

        for idx in batch_to_idx:
            n = len(idx)
            node_repr = torch.zeros(max_l, self.d_node).to(cryo_points_cube.device)
            node_repr[0 : n] = cryo_points_cube[idx]

            node_mask = torch.zeros(max_l).to(cryo_points_cube.device)
            node_mask[0 : n] = 1.0

            batch_node_repr.append(node_repr)
            batch_node_mask.append(node_mask)

        batch_node_repr = torch.stack(batch_node_repr, dim=0)
        batch_node_mask = torch.stack(batch_node_mask, dim=0)

        #print(batch_node_repr.shape)
        #print(batch_node_repr[0, :, 0])
        #print(batch_node_mask[0, :])
        #print(batch_node_repr[1, :, 0])
        #print(batch_node_mask[1, :])
   
        ###############
        # edge features
        ###############
        batch_edge_repr = []
        for b, idx in enumerate(batch_to_idx):
            n = len(idx)

            k = min(len(idx) - 1, self.k)
            bde_out = self.backbone_distance_emb(affines[idx], edge_index=None, batch=None, k=k)

            with torch.no_grad():
                batch_cryo_grids = [
                    cryo_grids[b].expand(n * k, -1, -1, -1, -1)
                ]

                cryo_vectors = bde_out.neighbour_positions.detach() # (n, k, d)

                cryo_vectors = [cryo_vectors.reshape(-1, 3)]

                cryo_vectors_center_positions = [
                    (
                        bde_out.positions
                        .unsqueeze(1)
                        .expand(n, k, 3)
                        .reshape(-1, 3)
                        - cryo_global_origins[b]
                    ) / cryo_voxel_sizes[b]
                ]

                cryo_vectors_rec = sample_centered_rectangle_along_vector(
                    batch_cryo_grids,
                    cryo_vectors,
                    cryo_vectors_center_positions,
                    rectangle_length=self.r_length,
                )  # (n k) 1 self.r_length 3 3

            neighbor_repr = self.conv_rectangle_pre(cryo_vectors_rec.requires_grad_()) # (n k) d self.r_length 1 1

            neighbor_repr = einops.rearrange(
                neighbor_repr, 
                "(b kz) c z y x -> b kz (c z y x)",
                kz=k,
                c=self.d_cryo_emb // 2,
                z=self.r_length,
                x=1,
                y=1,
            )

            neighbor_repr = self.conv_rectangle_post(neighbor_repr)

            edge_repr = torch.zeros(n, n, self.d_edge).to(neighbor_repr.device)

            edge_index = bde_out.edge_index

            row_idx = torch.arange(n).unsqueeze(1).expand(n, k)
            col_idx = edge_index

            edge_repr[row_idx, col_idx] = neighbor_repr

            edge_repr_x = torch.zeros(max_l, max_l, self.d_edge).to(edge_repr.device)

            edge_repr_x[:n, :n, :] = edge_repr

            batch_edge_repr.append(edge_repr_x)


        batch_edge_repr = torch.stack(batch_edge_repr, dim=0)

        return batch_node_repr, batch_edge_repr, batch_node_mask, batch_affines

import time
if __name__ == '__main__':
    model = CryoInit(k=64).cuda()
    print(sum(p.numel() for p in model.parameters()))

    l = 64

    affines = torch.randn(l, 3, 4).cuda()
    cryo_grids = [torch.ones(1, 1, 40, 40, 40).cuda()] * 2
    cryo_global_origins = [torch.zeros(3).cuda()] * 2
    cryo_voxel_sizes = [torch.ones(3).cuda()] * 2
    batch = torch.cat([
        torch.zeros(20).cuda(),
        torch.ones(44).cuda(),
    ], dim=-1).long()
    #batch = torch.zeros(64).cuda().long()
    

    for i in range(15):
        time.sleep(2)
        t0 = time.time()
        node, edge, mask, baffines = model(
            affines=affines,
            cryo_grids=cryo_grids,
            cryo_global_origins=cryo_global_origins,
            cryo_voxel_sizes=cryo_voxel_sizes,
            batch=batch,
        )
        t1 = time.time()
        print("{:.4f}".format(t1 - t0))
        print(node.shape)
        print(edge.shape)
        print(baffines.shape)

