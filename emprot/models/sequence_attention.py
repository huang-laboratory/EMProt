from collections import namedtuple, OrderedDict

import math
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from functools import partial

from emprot.utils.torch_utils import get_batch_slices, padded_sequence_softmax

def get_batched_sequence_attention_scores(
    sequence_query,  # n a f
    sequence_key,  # 1 s a f
    batch,  # n
    attention_scale,
    batch_size=200,
    device="cpu",
):
    output = torch.zeros(
        sequence_query.shape[0], *sequence_key.shape[1:3], device=device
    )  # n s a

    n_len, s_len = output.shape[:2]
    seq_batches = get_batch_slices(s_len, batch_size)
    sequence_query = sequence_query[:, None]

    for seq_batch in seq_batches:
        o = (sequence_query * sequence_key[:, seq_batch][batch]).sum(
            dim=-1
        ) / attention_scale

        output[:, seq_batch] = o

    return output

def get_batched_sequence_attention_features(
    sequence_attention_weights,  # n s a
    sequence_value,  # 1 s a f
    batch,  # n
    batch_size=200,
    device="cpu",
):
    output = torch.zeros(
        sequence_attention_weights.shape[0], *sequence_value.shape[2:], device=device
    )  # n a f
    n_len, s_len = sequence_attention_weights.shape[:2]
    seq_batches = get_batch_slices(s_len, batch_size)

    for seq_batch in seq_batches:
        output += (
            sequence_attention_weights[:, seq_batch][..., None]
            * sequence_value[:, seq_batch][batch]
        ).sum(dim=1)
    return output  # n a f

class SequenceAttention(nn.Module):
    def __init__(
        self,
        d: int = 256,
        d_seq: int = 1280,
        d_head: int = 48,
        n_head: int = 8,
        activation_class: nn.Module = nn.ReLU,
        checkpoint: bool = True,
    ):
        super().__init__()
        self.d = d
        self.d_seq = d_seq
        self.d_head = d_head
        self.n_head = n_head

        self.attention_scale = math.sqrt(self.d_head)
        self.norm = nn.LayerNorm(self.d)

        self.q = nn.Sequential(
            nn.Linear(self.d, self.n_head * self.d_head, bias=False),
            Rearrange(
                "... (h d) -> ... h d",
                h=self.n_head,
                d=self.d_head,
            ),
        )

        self.k = nn.Sequential(
            nn.Linear(self.d_seq, self.n_head * self.d_head, bias=False),
            Rearrange(
                "... (h d) -> ... h d",
                h=self.n_head,
                d=self.d_head,
            ),
        )

        self.v = nn.Sequential(
            nn.Linear(self.d_seq, self.n_head * self.d_head, bias=False),
            Rearrange(
                "... (h d) -> ... h d",
                h=self.n_head,
                d=self.d_head,
            ),
        )

        self.gate = nn.Sequential(
            nn.Linear(self.d, self.n_head * self.d_head, bias=True),
            Rearrange("... (h d) -> ... h d", h=self.n_head, d=self.d_head),
            nn.Sigmoid()
        )

        self.back = nn.Sequential(
            Rearrange("... h d -> ... (h d)", h=self.n_head, d=self.d_head),
            nn.Linear(self.n_head * self.d_head, self.d)
        )

        self.forward = self.forward_checkpoint if checkpoint else self.forward_normal

    def forward_normal(
        self,
        node,
        packed_sequence_emb,
        packed_sequence_mask,
        batch=None,
        attention_batch_size=200,
        **kwargs,
    ):
        return self._intern_forward(
            node, 
            packed_sequence_emb, 
            packed_sequence_mask, 
            batch, 
            attention_batch_size, 
        )

    def forward_checkpoint(
        self,
        node: torch.Tensor,
        packed_sequence_emb: torch.Tensor,
        packed_sequence_mask: torch.Tensor,
        batch=None,
        attention_batch_size: int = 200,
        **kwargs,
    ):
        new_forward = partial(
            self._intern_forward,
            packed_sequence_emb=packed_sequence_emb,
            packed_sequence_mask=packed_sequence_mask,
            batch=batch,
            attention_batch_size=attention_batch_size
        )
        return torch.utils.checkpoint.checkpoint(
            new_forward,
            node,
            preserve_rng_state=False,
        )

    def _intern_forward(
        self,
        node: torch.Tensor,
        packed_sequence_emb: torch.Tensor,
        packed_sequence_mask: torch.Tensor,
        batch,
        attention_batch_size: int,
    ):
        device = node.device
        if batch is None:
            batch = torch.zeros(node.shape[0], dtype=torch.long, device=device)

        # residual
        y = node

        # pre-norm
        y = self.norm(y)

        sequence_query = self.q(y)  # (n, h, d)
        sequence_key = self.k(packed_sequence_emb)  # (1, s, h, d)
        sequence_value = self.v(packed_sequence_emb)  # (1, s, h, d)

        sequence_attention_scores = get_batched_sequence_attention_scores(
            sequence_query,
            sequence_key,
            batch,
            self.attention_scale,
            batch_size=attention_batch_size,
            device=device,
        )

        batched_mask = packed_sequence_mask[batch].unsqueeze(-1)  # (n, s, 1)

        # Since sequence emb was padded, do not consider the padded parts for attention
        sequence_attention_weights = padded_sequence_softmax(
            sequence_attention_scores, batched_mask, dim=1
        )

        new_features_attention = get_batched_sequence_attention_features(
            sequence_attention_weights,
            sequence_value,
            batch,
            batch_size=attention_batch_size,
            device=device,
        )

        gate = self.gate(y)

        new_features = self.back(gate * new_features_attention)

        # residual connection
        new_features = node + new_features

        return new_features

import time
if __name__ == '__main__':
    device = "cuda:0"
    model = SequenceAttention(checkpoint=False).to(device)


    n = 192
    node = torch.randn(n,  256).cuda()
    seq_embed = torch.randn(1, 3000, 1280).cuda()
    seq_mask = torch.ones(1, 3000).bool().cuda()


    for i in range(10):
        start = time.time()
        x = model(
            node=node,
            packed_sequence_emb=seq_embed,
            packed_sequence_mask=seq_mask,
        )
        end = time.time()

        time.sleep(3)
        print(x.shape)
        print(x[0])

        print("{:.4f}".format(end - start))

