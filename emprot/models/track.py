import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

import time
import einops

from emprot.models.attention import NodeAttention, EdgeAttention
from emprot.models.ipa_pytorch import InvariantPointAttention, IPATransition, BackboneUpdate

# Components for three-track blocks
# 1. Single -> Single update (biased attention. bias from pair & structure)
# 2. Pair -> Pair update (biased attention. bias from structure)
# 3. Single -> Pair update (extract coevolution signal)
# 4. Str -> Str update


# this gives tensor of shape  (... l, l, 64) as distance map
def rbf(d, d_min=0.0, d_count=64, d_sigma=0.5):
    # Distance radial basis function
    d_max = d_min + (d_count - 1) * d_sigma
    d_mu = torch.linspace(d_min, d_max, d_count).to(d.device)
    d_mu = d_mu[None, :]
    d_expand = torch.unsqueeze(d, -1)
    rbf = torch.exp(-((d_expand - d_mu) / d_sigma) ** 2)
    return rbf

# TODO add tr-rosetta style orientation map 

class OutProduct(nn.Module):
    def __init__(
        self, 
        d_node=256, 
        d_edge=128, 
        d_head=32, 
        p_drop=0.15
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_node)
        self.left = nn.Linear(d_node, d_head)
        self.right = nn.Linear(d_node, d_head)
        self.out = nn.Linear(d_head * d_head, d_edge)

    def forward(self, node, edge):
        node = self.norm(node)

        left = self.left(node) # (..., n, d)
        right = self.right(node) # (..., n, d)

        # outer product
        out = torch.einsum("... n i, ... m j -> ... n m i j", left, right).flatten(-2) # (... n n d ** 2)
        out = self.out(out)

        out = out + edge
        return out


class TorsionNet(nn.Module):
    def __init__(
        self, 
        d_node=256, 
        d_head=128, 
        p_drop=0.15, 
        n_tors=10, 
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_node)
        self.linear_0 = nn.Linear(d_node, d_head)

        # ResNet layers
        self.linear_1 = nn.Linear(d_head, d_head)
        self.linear_2 = nn.Linear(d_head, d_head)
        self.linear_3 = nn.Linear(d_head, d_head)
        self.linear_4 = nn.Linear(d_head, d_head)

        # Final outputs
        self.n_tors = n_tors
        self.linear_out = nn.Linear(d_head, self.n_tors * 2)
    
    def forward(self, node):
        node = self.norm(node)
        node = self.linear_0(node)

        node = node + self.linear_2(F.relu_(self.linear_1(F.relu_(node))))
        node = node + self.linear_4(F.relu_(self.linear_3(F.relu_(node))))

        tors = self.linear_out(F.relu_(node))
        tors = einops.rearrange(tors, "... (d x) -> ... d x", d=self.n_tors, x=2)

        return tors


class TrackBlock(nn.Module):
    def __init__(
        self, 
        d_node=256, 
        d_edge=128, 
        d_bias=64,
        d_head=48, 
        n_head=8, 
        n_qk_point=4, 
        n_v_point=8, 
        p_drop=0.10,
    ):

        super().__init__()

        # node update
        self.node_update = NodeAttention(
            d_node=d_node,
            d_edge=d_edge, 
            d_bias=d_bias,
            d_head=d_head,
            n_head=n_head, 
        )

        # out product
        self.out_product = OutProduct(
            d_node=d_node,
            d_edge=d_edge,
            d_head=d_head // 2,
            p_drop=p_drop, 
        )

        # edge update
        self.edge_update = EdgeAttention(
            d_edge=d_edge,
            d_bias=d_bias,
            d_head=d_head,
            n_head=n_head,
            p_drop=p_drop,
        )


        # ipa
        self.ipa = InvariantPointAttention(
            d_node=d_node, 
            d_edge=d_edge, 
            d_head=d_head, 
            n_head=n_head, 
            n_qk_point=n_qk_point,
            n_v_point=n_v_point,
        )
        self.ipa_transition = IPATransition(d_node)
        self.bb_update = BackboneUpdate(d_node)

    def forward(
        self, 
        node, 
        edge, 
        affines, 
        node_pos=None, 
        edge_pos=None, 
        mask=None,
        use_checkpoint=False, 
    ):
        """
            node: (..., l, d)
            edge: (..., l, l, d)
            afines: (..., l, 3, 4)
        """

        if mask is None:
            mask = torch.ones_like(node[..., -1])

        edge_mask = mask[..., :, None] * mask[..., None, :] # (..., l, l)


        # embed distance map
        with torch.no_grad():
            positions = affines[..., :3, -1]
            distance = (positions[..., :, None, :] - positions[..., None, :, :]).norm(dim=-1)
            rbf_feat = rbf(distance)

            node_pos = affines[..., :3, -1] # (..., l, 3)
            edge_pos = node_pos[..., :, None, :] - node_pos[..., None, :, :]


        if use_checkpoint:
            raise NotImplementedError

        else:

            node = self.node_update(
                node, 
                edge, 
                rbf_feat, 
                node_pos, 
                mask=mask, 
            )

            edge = self.out_product(
                node,
                edge,
            )

            edge = self.edge_update(
                edge, 
                rbf_feat, 
                edge_pos, 
            )
            
            # no grad for rotation
            affines = torch.cat(
                [
                    affines[..., :3, :3].detach(), # no grad for rotation
                    affines[..., :3, -1][..., None], 
                ], 
                dim=-1, 
            )

            # bb update
            node = self.ipa(
                node, 
                edge, 
                affines, 
                node_pos,
                mask, 
            )
            node = self.ipa_transition(node)
            affines = self.bb_update(node, affines)

        return node, edge, affines

import time
if __name__ == '__main__':
    model = TrackBlock().cuda()
    print(sum(p.numel() for p in model.parameters()))

    for i in range(10):
        t0 = time.time()
        node, edge, affines = model(
            node=torch.randn(2, 10, 256).cuda(),
            edge=torch.randn(2, 10, 10, 128).cuda(),
            affines=torch.randn(2, 10, 3, 4).cuda(),
            node_pos=torch.randn(2, 10, 3).cuda(),
            edge_pos=torch.randn(2, 10, 10, 3).cuda(),
            mask=torch.ones(2, 10).cuda(),
            use_checkpoint=False,
        )
        t1 = time.time()
    
        print("{:.4f}".format(t1 - t0))

        time.sleep(2)
