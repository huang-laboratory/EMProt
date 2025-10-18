import math
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from emprot.models.rope import node_rope, edge_rope

class FeedForwardLayer(nn.Module):
    def __init__(
        self, 
        d=128, 
        n=2, 
        p_drop=0.1
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.linear1 = nn.Linear(d, d * n)
        self.dropout = nn.Dropout(p_drop)
        self.linear2 = nn.Linear(d * n, d)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear2(self.dropout(nn.functional.relu_(self.linear1(x))))
        return x

class NodeAttention(nn.Module):
    def __init__(
        self, 
        d_node: int = 256, 
        d_edge: int = 128,
        d_bias: int = 64, 
        d_head: int = 48, 
        n_head: int = 8, 
        inf: float = 1e6, 
    ):
        super().__init__()
        self.inf = inf

        self.norm_node = nn.LayerNorm(d_node)
        self.norm_edge = nn.LayerNorm(d_edge)

        self.bias = nn.Linear(d_bias, d_edge, bias=False)

        self.attn_scale = 1.0 / math.sqrt(d_node)

        self.q = nn.Sequential(
            nn.Linear(d_node, d_head * n_head, bias=False),
            Rearrange("... (h d) -> ... h d", h=n_head, d=d_head), 
        )
        self.k = nn.Sequential(
            nn.Linear(d_node, d_head * n_head, bias=False),
            Rearrange("... (h d) -> ... h d", h=n_head, d=d_head), 
        )
        self.v = nn.Sequential(
            nn.Linear(d_node, d_head * n_head, bias=False),
            Rearrange("... (h d) -> ... h d", h=n_head, d=d_head), 
        )

        self.b = nn.Linear(d_edge, n_head, bias=False)

        self.g = nn.Sequential(
            nn.Linear(d_node, n_head * d_head, bias=True),
            Rearrange("... (h d) -> ... h d", h=n_head, d=d_head),
            nn.Sigmoid(), 
        )

        self.o = nn.Sequential(
            Rearrange("... h d -> ... (h d)", h=n_head, d=d_head),
            nn.Linear(d_head * n_head, d_node, bias=False), 
        )

        self.residual = nn.Identity()

        self.ff = FeedForwardLayer(d_node, 2)

    # edge as bias, with extra bias
    def forward(self, node, edge, bias, node_pos, mask=None):
        if mask is None:
            mask = torch.ones_like(node[..., -1])

        # residue connect
        out_residual = self.residual(node)

        # norm edge
        edge = self.norm_edge(edge)
        edge = edge + self.bias(bias)

        # norm node
        node = self.norm_node(node)

        # attention
        q = self.q(node) # (..., l, h, d)
        k = self.k(node) # (..., l, h, d)
        v = self.v(node) # (..., l, h, d)

        q, k = node_rope(q, k, node_pos)
        gate = self.g(node) # (..., l, h, d)
        b = self.b(edge) # (..., l, l, h)

        q = q * self.attn_scale

        attn = torch.einsum('... i h d, ... j h d -> ... i j h', q, k)
        attn = attn + b

        # attn mask
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)
        attn = attn + square_mask.unsqueeze(-1)

        # get output
        attn = nn.functional.softmax(attn, dim=-2)
        out = torch.einsum('... i j h, b j h d -> ... i h d', attn, v)

        out = gate * out
        out = self.o(out)

        out = out_residual + self.ff(out)
        return out



class BiasedAxialAttention(nn.Module):
    def __init__(
        self, 
        d_edge=128, 
        d_bias=64, 
        n_head=4, 
        d_head=48,
        p_drop=0.1, 
        is_row=True,
        inf=1e6, 
    ):
        super().__init__()
        #
        self.inf = inf

        self.is_row = is_row
        self.norm_edge = nn.LayerNorm(d_edge)
        self.norm_bias = nn.LayerNorm(d_bias)
        self.attn_scale = 1.0 / math.sqrt(d_head)

        self.q = nn.Sequential(
            nn.Linear(d_edge, n_head * d_head, bias=False),
            Rearrange("... (h d) -> ... h d", h=n_head, d=d_head),
        )
        self.k = nn.Sequential(
            nn.Linear(d_edge, n_head * d_head, bias=False),
            Rearrange("... (h d) -> ... h d", h=n_head, d=d_head), 
        )
        self.v = nn.Sequential(
            nn.Linear(d_edge, n_head * d_head, bias=False),
            Rearrange("... (h d) -> ... h d", h=n_head, d=d_head),
        )

        self.b = nn.Linear(d_bias, n_head, bias=False)

        self.g = nn.Sequential(
            nn.Linear(d_edge, n_head * d_head, bias=True),
            Rearrange("... (h d) -> ... h d", h=n_head, d=d_head),
            nn.Sigmoid(), 
        )

        self.o = nn.Sequential(
            Rearrange("... h d -> ... (h d)", h=n_head, d=d_head),
            nn.Linear(d_head * n_head, d_edge, bias=False), 
        )

    def forward(self, edge, bias, edge_pos, edge_mask=None):
        B, L = edge.shape[:2]
        
        if edge_mask is None:
            edge_mask = torch.ones_like(edge[..., -1])

        if self.is_row:
            edge = edge.permute(0, 2, 1, 3)
            bias = bias.permute(0, 2, 1, 3)

        edge = self.norm_edge(edge)
        bias = self.norm_bias(bias)

        q = self.q(edge)
        k = self.k(edge)
        v = self.v(edge) # (b, l, l, h, d)
        q, k = edge_rope(q, k, edge_pos)

        bias = self.b(bias) # (b, l, l, h)

        gate = torch.sigmoid(self.g(edge)) # (b, l, l, h * d)

        q = q * self.attn_scale
        k = k / L

        attn = torch.einsum('b n i h k, b n j h k -> b i j h', q, k) # tied attention
        attn = attn + bias # apply bias
        attn = nn.functional.softmax(attn, dim=-2) # (b, l, l, h)

        # TODO add mask

        # add mask
        edge_mask = self.inf * (edge_mask - 1) # (b, l, l)
        attn = attn + edge_mask.unsqueeze(-1)

        # get output
        out = torch.einsum('b i j h, b n j h d -> b n i h d', attn, v)
        out = gate * out

        out = self.o(out)

        if self.is_row:
            out = out.permute(0,2,1,3)

        return out


class Dropout(nn.Module):
    # Dropout entire row or column
    def __init__(
        self, 
        broadcast_dim=None, 
        p_drop=0.10,
    ):
        super().__init__()
        # give ones with probability of 1-p_drop / zeros with p_drop
        self.sampler = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-p_drop]))
        self.broadcast_dim=broadcast_dim
        self.p_drop=p_drop

    def forward(self, x):
        if not self.training: # no drophead during evaluation mode
            return x
        shape = list(x.shape)
        if not self.broadcast_dim == None:
            shape[self.broadcast_dim] = 1
        mask = self.sampler.sample(shape).to(x.device).view(shape)
        x = mask * x / (1.0 - self.p_drop)
        return x


class EdgeAttention(nn.Module):
    def __init__(
        self, 
        d_edge: int = 128, 
        d_bias: int = 64, 
        d_head: int = 48, 
        n_head: int = 4,
        p_drop: float = 0.10, 
    ):
        super().__init__()

        self.drop_row = Dropout(broadcast_dim=1, p_drop=p_drop)
        self.drop_col = Dropout(broadcast_dim=2, p_drop=p_drop)

        self.row_attn = BiasedAxialAttention(d_edge, d_bias, n_head, d_head, p_drop=p_drop, is_row=True)
        self.col_attn = BiasedAxialAttention(d_edge, d_bias, n_head, d_head, p_drop=p_drop, is_row=False)

        self.ff = FeedForwardLayer(d_edge, 2)

    def forward(self, edge, edge_bias, edge_pos, edge_mask=None):
        edge = edge + self.drop_row(self.row_attn(edge, edge_bias, edge_pos, edge_mask=edge_mask))
        edge = edge + self.drop_col(self.col_attn(edge, edge_bias, edge_pos, edge_mask=edge_mask))
        edge = edge + self.ff(edge)
        return edge




class EdgeTransition(nn.Module):
    def __init__(
        self,
        d_node=256,
        d_edge=128,
        d_bias=64, 
        n_layer=2,
        p_drop=0.10, 
    ):
        super().__init__()

        self.residual = nn.Identity()
        self.norm_edge = nn.LayerNorm(d_edge)

        self.initial_embed = nn.Linear(d_node, d_bias)
        d_hidden = d_bias * 3 + d_edge

        trunk_layers = []
        for _ in range(n_layer):
            trunk_layers.append(nn.Linear(d_hidden, d_hidden))
            trunk_layers.append(nn.ReLU())

        self.trunk = nn.Sequential(*trunk_layers)
        self.final_layer = nn.Linear(d_hidden, d_edge)

        self.drop_row = Dropout(broadcast_dim=1, p_drop=p_drop)
        self.drop_col = Dropout(broadcast_dim=2, p_drop=p_drop)

        self.ff = FeedForwardLayer(d_edge, 2)

    def forward(self, node, edge, edge_bias):
        b, l, d = node.shape
        edge_initial = self.residual(edge)

        edge_embed = self.norm_edge(edge)

        node_embed = self.initial_embed(node)

        bias = torch.cat([
            torch.tile(node_embed[:, :, None, :], (1, 1, l, 1)),
            torch.tile(node_embed[:, None, :, :], (1, l, 1, 1)),
        ], axis=-1)

        edge_embed = torch.cat([
            edge_embed, bias, edge_bias
        ], axis=-1)

        edge_embed = self.trunk(edge_embed)

        edge_embed = self.final_layer(edge_embed)

        edge_embed = edge_embed + self.drop_row(edge_embed)
        edge_embed = edge_embed + self.drop_col(edge_embed)
        edge_embed = edge_embed + self.ff(edge_embed)

        edge_embed = edge_initial + edge_embed

        return edge_embed



import time
if __name__ == '__main__':
    model = EdgeAttention().cuda()
    print(sum([p.numel() for p in model.parameters()]))

    edge = torch.rand(2, 128, 128, 128).cuda()
    bias = torch.rand(2, 128, 128, 64).cuda()
    edge_pos = torch.rand(2, 128, 128, 3).cuda()
    edge_mask = torch.ones(2, 128, 128).float().cuda()

    for i in range(10):
        t0 = time.time()
        x = model(edge, bias, edge_pos, edge_mask=edge_mask)
        t1 = time.time()
        print("{:.4f}".format(t1 - t0))

    """
    model = EdgeTransition().cuda()
    print(sum([p.numel() for p in model.parameters()]))

    node = torch.rand(2, 128, 256).cuda()
    edge = torch.rand(2, 128, 128, 128).cuda()
    edge_bias =  torch.rand(2, 128, 128, 64).cuda()
    for i in range(10):
        t0 = time.time()
        x = model(node, edge, edge_bias)
        t1 = time.time()
        print("{:.4f}".format(t1 - t0))
    """
 

    model = NodeAttention().cuda()
    print(sum([p.numel() for p in model.parameters()]))

    node = torch.rand(2, 128, 256).cuda()
    edge = torch.rand(2, 128, 128, 128).cuda()
    bias = torch.rand(2, 128, 128, 64).cuda()
    node_pos = torch.rand(2, 128, 3).cuda()
    mask = torch.ones(2, 128).float().cuda()

    for i in range(10):
        t0 = time.time()
        x = model(node, edge, bias, node_pos, mask=mask)
        t1 = time.time()
        print("{:.4f}".format(t1 - t0))
