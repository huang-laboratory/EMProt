import torch
import torch.nn as nn
from typing import List
import contextlib

from emprot.models.cryo_init import CryoInit
from emprot.models.track import TrackBlock, TorsionNet

class Output:
    def __init__(self, **kwargs):
        self.data = dict()
        for k, v in kwargs.items():
            self.data[k] = v

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __contains__(self, key):
        return key in self.data

    def to(self, device_or_dtype):
        for k, v in self.data.items():
            if torch.is_tensor(v):
                self.data[k] = v.to(device_or_dtype)
            elif isinstance(v, list):
                self.data[k] = [x.to(device_or_dtype) for x in v]
        return self
        

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def update(self, **kwargs):
        self.data.update(kwargs)

    def __repr__(self):
        return f"Output({self.data})"



class Model(nn.Module):
    def __init__(
        self, 
        d_node=256,
        d_edge=128,
        d_bias=64,
        d_head=48,
        n_qk_point=4,
        n_v_point=8, 
        n_head=8,
        n_block=8, 
        n_tors=10, 
        k=64,
        p_drop=0.10,
        use_checkpoint=False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        # Init rel pos
        self.max_pos = 32
        self.embed_rel_pos = nn.Embedding(65, d_edge)
        self.embed_same_chain = nn.Embedding(2, d_edge)

        # Feature init
        self.init = CryoInit(
            d_node=d_node,
            d_edge=d_edge,
            d_cryo_emb=d_node,
            k=k,
        )

        # Track blocks
        self.n_block = n_block
        self.blocks = nn.ModuleList()
        for i in range(self.n_block):
            self.blocks.append(TrackBlock(
                d_node=d_node,
                d_edge=d_edge,
                d_bias=d_bias,
                d_head=d_head,
                n_head=n_head,
                n_qk_point=n_qk_point,
                n_v_point=n_v_point,
                p_drop=p_drop,
            ))

        # Torsions
        self.n_tors = n_tors
        self.torsion_predictor = TorsionNet(d_node, d_node, n_tors=self.n_tors)

    def forward(
        self, 
        affines: torch.Tensor, 
        cryo_grids: List[torch.Tensor],
        cryo_global_origins: List[torch.Tensor],
        cryo_voxel_sizes: List[torch.Tensor], 
        batch=None, 
        run_iters=1, 
        residue_index=None,
        chain_index=None,
        **kwargs, 
    ):
        # Embed rel pos
        same_chain = chain_index[..., :, None] == chain_index[..., None, :]
        same_chain = same_chain.long()
        same_chain_embed = self.embed_same_chain(same_chain)

        residue_index = chain_index * int(1e5) + residue_index
        rel_pos = residue_index[..., :, None] - residue_index[..., None, :]
        rel_pos = torch.clip(rel_pos, min=-self.max_pos, max=self.max_pos) + self.max_pos
        rel_pos = rel_pos.long()
        rel_pos_embed = self.embed_rel_pos(rel_pos)

        #print(same_chain)
        #print(rel_pos)

        # batchify is still buggy for b > 1 in training
        # n = 1 is good

        if batch is None:
            batch = torch.zeros( len(affines) ).to(affines.device).long()

        init_affines = affines

        for run_iter in range(run_iters):
            with torch.no_grad() if run_iter < run_iters - 1 else contextlib.nullcontext():
                # Init features from map
                batch_node, batch_edge, batch_node_mask, batch_affines = self.init(
                    affines=init_affines,
                    cryo_grids=cryo_grids, 
                    cryo_global_origins=cryo_global_origins, 
                    cryo_voxel_sizes=cryo_voxel_sizes, 
                    batch=batch,
                )

                # Add positional encoding
                #print(same_chain_embed.shape)
                #print(rel_pos_embed.shape)
                #print(batch_edge.shape)
                batch_edge = batch_edge + same_chain_embed[None, ...] + rel_pos_embed[None, ...] # add batch dim

                #print(batch_node.shape)
                #print(batch_edge.shape)
                #print(batch_node_mask.shape)
                #print(batch_affines.shape)

                affines_list = []
                for i in range(self.n_block):
                    # main blocks
                    batch_node, batch_edge, batch_affines = self.blocks[i](
                        batch_node, batch_edge, batch_affines, 
                        mask=batch_node_mask, 
                        use_checkpoint=self.use_checkpoint, 
                    )

                    # convert batch affines
                    affines = batch_affines[batch_node_mask.bool()]
                    affines_list.append(affines)

                    #print(affines.shape)

                # Torsion prediction
                torsions = self.torsion_predictor(batch_node)
                torsions = torsions[batch_node_mask.bool()]

                #print(torsions.shape)

                init_affines = affines_list[-1]

        return Output(
            pred_torsions=torsions, 
            pred_affines=affines_list, 
            pred_positions=[affines[..., :3, -1] for affines in affines_list],
        )

import time
if __name__ == '__main__':
    model = Model(n_block=8).cuda()
    print(sum(p.numel() for p in model.parameters()))

    for i in range(10):
        t0 = time.time()

        out = model(
            affines=torch.randn(128, 3, 4).cuda(),
            cryo_grids=[torch.zeros(1, 1, 120, 120, 120).cuda()] ,
            cryo_global_origins=[torch.zeros(3).cuda()] ,
            cryo_voxel_sizes=[torch.ones(3).cuda()] ,
            residue_index=torch.arange(0, 128).cuda(),
            chain_index=torch.zeros(128).cuda(),
            run_iters=2,
        )
        t1 = time.time()

        print("{:.4f}".format(t1 - t0))

        #time.sleep(1)
