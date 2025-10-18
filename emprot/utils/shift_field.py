import numpy as np

def dist2(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    assert a.ndim == b.ndim
    assert a.shape[-1] == b.shape[-1]
    distances = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1) # (L, M)
    return distances

def create_shift_field(fixing, moving, u=15.0):
    # fixing and moving of shape (M, 3)
    assert fixing.shape == moving.shape
    shift_vectors = moving - fixing # (M, 3)
    def smoothed_shift_field(x):
        # x of shape (L, 3)
        assert x.ndim == 2
        x = np.asarray(x, dtype=np.float32) # (L, 3)
        weights = np.exp(-dist2(x, fixing) / u**2) # (L, M)
        weighted_shifts = weights @ shift_vectors # (L, M) @ (M, 3) -> (L, 3)
        sum_weights = np.sum(weights, axis=-1) # (L, )
        return weighted_shifts / (sum_weights[..., None] + 1e-6)
    return smoothed_shift_field


if __name__ == '__main__':
    ca_coords1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    ca_coords2 = np.array([[1.1, 2.1, 2.8], [4.1, 5.0, 6.2], [6.8, 7.8, 9.2]])
    
    u = 15.0
    shift_field = create_shift_field(ca_coords1, ca_coords2, u)
   
    test_point = np.array([[2.5, 3.5, 4.5]])
    smoothed_shift = shift_field(test_point)
    print("Smoothed shift at point:", smoothed_shift)
    print("Before shift:", test_point)
    test_point += smoothed_shift
    print("After  shift:", test_point)
