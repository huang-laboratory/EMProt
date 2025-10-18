import numpy as np

def corr_np(x, y, eps=1e-8):
    assert x.shape == y.shape
    x = x - x.mean()
    y = y - y.mean()
    numerator = np.sum(x * y)
    denominator = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
    return numerator / (denominator + eps)

def corr(x, y, eps=1e-8):
    assert x.shape == y.shape
    x = x - x.mean()
    y = y - y.mean()
    numerator = (x * y).sum()
    denominator = ((x ** 2).sum() * (y ** 2).sum()).sqrt()
    return numerator / (denominator + eps)

