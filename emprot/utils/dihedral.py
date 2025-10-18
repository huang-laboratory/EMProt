import torch

def get_dihedral(p0, p1, p2, p3, eps=1e-8):
    """
    Given p0-p3, compute dihedral b/t planes p0p1p2 and p1p2p3.
    """
    assert p0.shape[-1] == p1.shape[-1] == p2.shape[-1] == p3.shape[-1] == 3

    # dx
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    # normals
    n012 = torch.cross(b0, b1)
    n123 = torch.cross(b1, b2)

    # dihedral
    cos_theta = torch.einsum('...i,...i->...', n012, n123) / (
            torch.norm(n012, dim=-1) * torch.norm(n123, dim=-1) + eps)
    sin_theta = torch.einsum('...i,...i->...', torch.cross(n012, n123), b1) / (
            torch.norm(n012, dim=-1) * torch.norm(n123, dim=-1) * torch.norm(b1, dim=-1) + eps)
    #theta = torch.atan2(sin_theta, cos_theta)
    sin_cos_theta = torch.cat([sin_theta, cos_theta], dim=-1)

    return sin_cos_theta
