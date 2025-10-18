from collections import namedtuple
import numpy as np
import torch
from torch import nn

from emprot.utils.torch_utils import get_batches_to_idx
from emprot.utils.affine_utils import vecs_to_local_affine
from emprot.utils.knn_graph import knn_graph

BackboneDistanceEmbeddingOutput = namedtuple(
    "BackboneDistanceEmbeddingOutput",
    [
        "positions",
        "neighbour_positions",
        "edge_index",
        "full_edge_index",
    ],
)

class BackboneDistanceEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, 
        affines, 
        edge_index = None, 
        batch = None,
        k = 32, 
    ):
        positions = affines[..., :3, -1] # (n, 3)

        if edge_index is None:
            edge_index = knn_graph(positions, k, batch=batch, loop=False, flow="source_to_target")
            full_edge_index = edge_index
            edge_index = edge_index[0].reshape(len(positions), k) # n k

        neighbour_positions = vecs_to_local_affine(affines, positions[edge_index]) # n k

        return BackboneDistanceEmbeddingOutput(
            positions=positions,
            neighbour_positions=neighbour_positions,
            edge_index=edge_index,
            full_edge_index=full_edge_index,
        )
