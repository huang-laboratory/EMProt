# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modified code of Openfold's IPA."""

import numpy as np
import torch
import math
from scipy.stats import truncnorm
import torch.nn as nn
from typing import Optional, Callable, List, Sequence

from emprot.models.rope import node_rope
from emprot.models.rigid_utils import Rigid, Rotation

from emprot.utils.affine_utils import (
    affine_mul_vecs,
    affine_composition,
    get_affine, 
    quaternion_to_matrix, 
)

def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)

def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f

def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


class InvariantPointAttention(nn.Module):
    def __init__(
        self,
        d_node: int = 256,
        d_edge: int = 128,
        n_head: int = 8,
        d_head: int = 48,
        n_qk_point: int = 4,
        n_v_point: int = 8,
        eps: float = 1e-6,
        inf: float = 1e6,
    ):
        super().__init__()

        self.d_node = d_node
        self.d_edge = d_edge

        self.n_head = n_head
        self.d_head = d_head

        self.n_qk_point = n_qk_point
        self.n_v_point = n_v_point

        self.inf = inf
        self.eps = eps

        self.q = nn.Linear(d_node, n_head * d_head)
        self.kv = nn.Linear(d_node, n_head * d_head * 2)

        hpq = n_head * n_qk_point * 3
        self.q_points = nn.Linear(d_node, hpq)
        hpkv = n_head * (n_qk_point + n_v_point) * 3
        self.kv_points = nn.Linear(d_node, hpkv)

        self.bias = nn.Linear(d_edge, n_head)
        self.down_edge = nn.Linear(d_edge, d_edge // 4)

        self.head_weights = nn.Parameter(torch.zeros((n_head)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim =  (
            d_edge // 4 + d_head + n_v_point * 4
        )
        self.out = nn.Linear(n_head * concat_out_dim, d_node)

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def to_rigids(self, affines):
        return Rigid(Rotation(affines[..., :3, :3]), affines[..., :3, -1])

    def to_affines(self, rigids):
        return torch.cat([
            rigids.get_rots().get_rot_mats(),
            rigids.get_trans()[..., None], 
        ], dim=-1)

    def forward(self, s, z, affines, node_pos, mask=None):
        """
            s:
                [*, l, d] node repr.
            z:
                [*, l, l, d] edge repr.
            affines:
                [*, l, 3, 4] affines
        """

        if mask is None:
            mask = torch.ones_like(s[..., 0]) # [*, l]

        # convert affines to rigids
        r = self.to_rigids(affines)

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, l, h * d]
        q = self.q(s)
        kv = self.kv(s)

        # [*, l, h, d]
        q = q.view(q.shape[:-1] + (self.n_head, -1))

        # [*, l, h, 2 * d]
        kv = kv.view(kv.shape[:-1] + (self.n_head, -1))

        # [*, l, h, d] for each
        k, v = torch.split(kv, self.d_head, dim=-1)

        # rope
        q, k = node_rope(q, k, node_pos)

        # [*, l, h * qp * 3]
        q_pts = self.q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [*, l, h * qp, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)

        q_pts = r[..., None].apply(q_pts)

        # [*, l, h, qp, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.n_head, self.n_qk_point, 3)
        )

        # [*, l, h * (qp + vp) * 3]
        kv_pts = self.kv_points(s)

        # [*, l, h * (qp + vp), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)

        # [*, l, h, (qp + vp), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.n_head, -1, 3))

        # [*, l, h, qp / vp, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.n_qk_point, self.n_v_point], dim=-2
        )

        ##########################
        # Compute attention scores
        ##########################
        # [*, l, l, h]
        b = self.bias(z)

        # TODO
        # add neighbor mask here

        # [*, h, l, l]
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, h, l, d_head]
            permute_final_dims(k, (1, 2, 0)),  # [*, h, d_head, l]
        )
        a *= math.sqrt(1.0 / (3 * self.d_head))
        a += (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))

        # [*, l, l, h, qp, 3]
        pt_displacement = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_displacement ** 2

        # [*, l, l, h, qp]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.n_qk_point * 9.0 / 2))
        )
        pt_att = pt_att * head_weights

        # [*, l, l, h]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, l, l]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, h, l, l]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        a = a + pt_att
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, l, h, d_head]
        o = torch.matmul(
            a, v.transpose(-2, -3)
        ).transpose(-2, -3)

        # [*, l, h * d_head]
        o = flatten_final_dims(o, 2)

        # [*, h, 3, l, vp]
        o_pt = torch.sum(
            (
                a[..., None, :, :, None]
                * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
            ),
            dim=-2,
        )

        # [*, l, h, vp, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, l, h * vp]
        o_pt_dists = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(
            o_pt_dists, 2
        )

        # [*, l, h * vp, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        # [*, l, h, z // 4]
        pair_z = self.down_edge(z)
        o_pair = torch.matmul(a.transpose(-2, -3), pair_z)

        # [*, l, h * z // 4]
        o_pair = flatten_final_dims(o_pair, 2)

        o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair]

        # [*, l, s]
        s = self.out(
            torch.cat(
                o_feats, dim=-1
            )
        )

        return s


class IPATransition(nn.Module):
    def __init__(self, d: int = 256):
        super().__init__()

        self.norm = nn.LayerNorm(d)
        self.linear_1 = nn.Linear(d, d, bias=False)
        self.linear_2 = nn.Linear(d, d, bias=False)
        self.linear_3 = nn.Linear(d, d, bias=False)
        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.norm(s)
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)
        s = s + s_initial
        return s


class BackboneUpdate(nn.Module):
    def __init__(self, d: int=256):
        super().__init__()
        self.backbone_fc = nn.Linear(d, 6)
        self.f1 = nn.Parameter(torch.tensor(1.5, dtype=torch.float))
        self.eps = 1e-6

        self.init_parameters()

    def init_parameters(self):
        torch.nn.init.normal_(self.backbone_fc.weight, std=0.02)
        self.backbone_fc.bias.data = torch.Tensor([0, 0, 0, 0, 0, 0])

    def forward(self, x, affines):
        y = self.backbone_fc(x) # (..., 6)

        y = torch.cat(
            [
                torch.sqrt(torch.square(self.f1) + self.eps) * torch.ones(size=y.shape[:-1] + (1,)).float().to(y.device),
                y,
            ],
            dim=-1,
        )

        # To rotmats and trans
        quats = y[..., :4]
        trans = y[..., 4:]
        rotmats = quaternion_to_matrix(quats)
        new_affines = affine_composition(
            affines,
            get_affine(rotmats, trans),
        ) # (..., 3, 4)

        return new_affines



import time
if __name__ == '__main__':
    ipa = InvariantPointAttention().cuda()
    trans = IPATransition().cuda()
    bb_update = BackboneUpdate().cuda()

    print(sum([p.numel() for p in ipa.parameters()]))
    print(sum([p.numel() for p in trans.parameters()]))
    print(sum([p.numel() for p in bb_update.parameters()]))

    l = 10
    s = torch.randn(2, l, 256).cuda()
    z = torch.randn(2, l, l, 128).cuda()
    affines = torch.randn(2, l, 3, 4).cuda()
    mask = torch.ones((2, l)).cuda()
    mask[..., 5:10] = 0.0
 
    node_pos = torch.randn(2, l, 3).cuda()

    for i in range(10):
        t0 = time.time()
        s = ipa(
            s, z, affines, node_pos, mask=mask, 
        )
        new_affines = bb_update(s, affines)
        s = trans(s)

        #print(s.shape)
        #print(s[0, 0])
        #print(affines[0, 0])
        #print(new_affines[0, 0])

        t1 = time.time()
        print("{:.4f}".format(t1 - t0))
