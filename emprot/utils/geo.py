import os
import numpy as np

def split_chain_to_frags(
    ca_coords: np.ndarray,
    threshold: float = 8.0,
) -> list[list[int]]:
    ca_coords = np.asarray(ca_coords)
    dists = np.linalg.norm(ca_coords[1:] - ca_coords[:-1], axis=1)
    breaks = np.where(dists > threshold)[0] + 1

    segments = []
    start = 0
    for end in breaks:
        segments.append(list(range(start, end)))
        start = end
    segments.append(list(range(start, len(ca_coords))))

    return segments

def distance(a, b):
    # a, b of shape (3, )
    return np.linalg.norm(a-b)

def pairwise_distances(a, b):
    """
    a, b : np.ndarray of shape [L, d], [M, d]
    """
    assert a.ndim == b.ndim
    assert a.shape[-1] == b.shape[-1]
    distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return distances

def rmsd(P, Q):
    """
    P, Q of shape (L, 3)
    """
    return np.sqrt(
        np.mean(
            np.sum((P - Q) ** 2, axis=-1), # (..., L)
            axis=-1, # (..., )
        )
    )


def kabsch(P, Q):
    P = np.asarray(P, dtype=np.float32)
    Q = np.asarray(Q, dtype=np.float32)
    assert len(P) == len(Q)
    assert len(P) > 0

    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    H = P_centered.T.dot(Q_centered)
    U, S, VT = np.linalg.svd(H)
    R = U.dot(VT).T
    if np.linalg.det(R) < 0:
        VT[2,:] *= -1
        R = U.dot(VT).T
    t = centroid_Q - R.dot(centroid_P)

    return  R, t

def kabsch_rmsd(P, Q):
    P = np.asarray(P, dtype=np.float32)
    Q = np.asarray(Q, dtype=np.float32)

    R, T = kabsch(P, Q)
    #rP = P @ R.T + T
    axes = tuple(range(R.ndim - 2)) + (-1, -2)
    rP = np.matmul(P, R.transpose(axes)) + T
    
    #print(rP)
    #print(Q)

    #return np.sqrt(np.sum((rP - Q) ** 2))
    return np.sqrt(
        np.mean(
            np.sum((rP - Q) ** 2, axis=-1), # (..., L)
            axis=-1, # (..., )
        )
    )


# Vectorized batch operation
def kabschx(P, Q):
    assert P.shape == Q.shape, "Shapes of input tensors P and Q must be the same"
    assert len(P.shape) >= 2 and P.shape[-1] == 3, "Input tensors must have shape (..., L, 3)"

    # Calculate centroid of P and Q
    centroid_P = np.mean(P, axis=-2, keepdims=True)
    centroid_Q = np.mean(Q, axis=-2, keepdims=True)

    # Centered coordinates
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    # Compute covariance matrix H
    axes = tuple(range(P_centered.ndim - 2)) + (-1, -2)
    H = np.matmul(P_centered.transpose(axes), Q_centered)

    # Singular Value Decomposition of H
    U, S, VT = np.linalg.svd(H)

    # Compute rotation matrix R
    R = np.matmul(U, VT)
    axes = tuple(range(R.ndim - 2)) + (-1, -2)
    R = R.transpose(axes)

    # Ensure proper rotation matrix (det(R) = +1)
    det = np.linalg.det(R)
    sign = np.where(det < 0.0, -1, 1)[..., None]
    VT[..., 2, :] *= sign
    R = np.matmul(U, VT)
    R = R.transpose(axes)

    # Compute translation vector t
    axes = tuple(range(P_centered.ndim - 2)) + (-1, -2)
    t = centroid_Q - np.matmul(R, centroid_P.transpose(axes)).transpose(axes)

    return R, t


def kabschx_apply(P, R, t):
    axes = tuple(range(R.ndim - 2)) + (-1, -2)
    rP = np.matmul(P, R.transpose(axes)) + t
    return rP

def kabschx_rmsd(P, Q):
    P = np.asarray(P, dtype=np.float32)
    Q = np.asarray(Q, dtype=np.float32)

    R, T = kabschx(P, Q)
    axes = tuple(range(R.ndim - 2)) + (-1, -2)
    rP = np.matmul(P, R.transpose(axes)) + T
    return np.sqrt(
        np.mean(
            np.sum((rP - Q) ** 2, axis=-1), # (..., L)
            axis=-1, # (..., )
        )
    )
    
# Apply a transformation
def apply(x, R, t):
    # x (..., 3)
    # R (..., 3, 3)
    # t (..., 3)
    return kabschx_apply(x, R, t)



#################################################
###### Rotations ################################
#################################################
def euler_to_rot_mat(angle):
    """ZXZ style in https://zhuanlan.zhihu.com/p/607015899"""
    rot_mat = np.zeros((3, 3), dtype=np.float32)
    phi = angle[0]
    the = angle[1]
    psi = angle[2]
    
    cpsi = np.cos(psi)
    spsi = np.sin(psi)
    cthe = np.cos(the)
    sthe = np.sin(the)
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    
    rot_mat[0, 0] =  cpsi * cphi - cthe * sphi * spsi
    rot_mat[1, 0] =  cpsi * sphi + cthe * cphi * spsi
    rot_mat[2, 0] =  spsi * sthe
    
    rot_mat[0, 1] = -spsi * cphi - cthe * sphi * cpsi
    rot_mat[1, 1] = -spsi * sphi + cthe * cphi * cpsi
    rot_mat[2, 1] =  cpsi * sthe
    
    rot_mat[0, 2] =  sthe * sphi
    rot_mat[1, 2] = -sthe * cphi
    rot_mat[2, 2] =  cthe

    return rot_mat


def rot_mat_to_euler(rot_mat):
    """Ignore gimbal lock"""
    angle = np.zeros(3, dtype=np.float32)
    angle[0] = np.arctan2(rot_mat[0, 2], -rot_mat[1, 2])
    angle[1] = np.arccos(rot_mat[2, 2])
    angle[2] = np.arctan2(rot_mat[2, 0],  rot_mat[2, 1])
    return angle

def random_euler_safe():
    phi = np.random.uniform(-np.pi, np.pi)
    theta = np.random.uniform(-np.pi/3, np.pi/3)
    psi = np.random.uniform(-np.pi, np.pi)
    return np.array([phi, theta, psi])

def random_rot_mat_from_euler():
    euler = random_euler_safe()
    return euler_to_rot_mat(euler)




def random_quaternion():
    q = np.random.randn(4)
    q /= np.linalg.norm(q)  # normalize
    return q

def quaternion_to_rot_mat(q):
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    R = np.array([
        [1 - 2*q2**2 - 2*q3**2, 2*q1*q2 - 2*q3*q0, 2*q1*q3 + 2*q2*q0],
        [2*q1*q2 + 2*q3*q0, 1 - 2*q1**2 - 2*q3**2, 2*q2*q3 - 2*q1*q0],
        [2*q1*q3 - 2*q2*q0, 2*q2*q3 + 2*q1*q0, 1 - 2*q1**2 - 2*q2**2]
    ])
    return R


def random_rot_mat_from_quaternion():
    q = random_quaternion()
    return quaternion_to_rot_mat(q)



if __name__ == '__main__':
    a = np.array([[1,4,7], [2,5,8]])
    b = np.array([[1,2,1], [2,5,2]])

    #R, T = kabschx(a[None], b[None])
    print(R)
    print(T)

    R, T = kabsch(a, b)
    print(R)
    print(T)

    rmsd = kabsch_rmsd(a, b)
    print(rmsd)   
 
    R, T = kabschx(
        np.stack([a, a], axis=0), 
        np.stack([b, b], axis=0),
    )
    print(R)
    print(T)

    rmsd = kabschx_rmsd(
        np.stack([a, a], axis=0),
        np.stack([b, b], axis=0),
    )
    print(rmsd)
