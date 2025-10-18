import torch
import torch.nn as nn
import einops

def sinusoidalpositionalencoding(x, d=16, freq_base=1000):
    inv_freq = 1.0 / (freq_base ** (torch.arange(0, d, 2).float() / d))
    sin_inp_x = torch.einsum("... i, j -> ... i j", x, inv_freq.to(x.device))
    return sin_inp_x.cos(), sin_inp_x.sin() # (..., 3, d // 2)

def rotate(x, cos, sin):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rot

def node_rope(q, k, node_pos, freq_base=1000):
    d = q.shape[-1] 
    assert d % 3 == 0, "d must be % 3 == 0"
    assert d % 2 == 0, "d must be % 2 == 0"

    cos, sin = sinusoidalpositionalencoding(node_pos, d // 3)
    cos = cos.flatten(-2)
    sin = sin.flatten(-2)
    cos = cos[..., None, :]
    sin = sin[..., None, :]

    return rotate(q, cos, sin), rotate(k, cos, sin)


def edge_rope(q, k, edge_pos, freq_base=1000):
    b, n, nk, h, d = q.shape
    assert d % 3 == 0, "d must be % 3 == 0"
    assert d % 2 == 0, "d must be % 2 == 0"

    q_r = einops.rearrange(q, "b n k h d -> b (n k) h d", b=b, n=n, k=nk, h=h, d=d)
    k_r = einops.rearrange(k, "b n k h d -> b (n k) h d", b=b, n=n, k=nk, h=h, d=d)
    edge_pos_r = einops.rearrange(edge_pos, "b n k d -> b (n k) d", b=b, n=n, k=nk, d=3)
    
    q_r, k_r = node_rope(q_r, k_r, edge_pos_r)

    q = einops.rearrange(q_r, "b (n k) h d -> b n k h d", b=b, n=n, k=nk, h=h, d=d)
    k = einops.rearrange(k_r, "b (n k) h d -> b n k h d", b=b, n=n, k=nk, h=h, d=d)

    return q, k


if __name__ == '__main__':
    b = 2
    n = 20
    k = 20
    h = 4
    d = 48

    q = torch.rand(b, n, h, d)
    k = torch.rand(b, n, h, d)
    node_pos = torch.rand(b, n, 3)

    q, k = node_rope(q, k, node_pos)

    print(q.shape)
    print(k.shape)
    

    q = torch.rand(b, n, n, h, d)
    k = torch.rand(b, n, n, h, d)
    edge_pos = torch.rand(b, n, n, 3)

    q, k = edge_rope(q, k, edge_pos)

    print(q.shape)
    print(k.shape)


