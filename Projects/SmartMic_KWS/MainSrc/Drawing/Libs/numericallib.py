import numpy as np
from functools import reduce

def norm_signal(frames):
    return frames/np.max(frames)

def dct(n_filters, n_input):
    basis = np.empty((n_filters, n_input))
    basis[0, :] = 1.0 / np.sqrt(n_input)

    samples = np.arange(1, 2 * n_input, 2) * np.pi / (2.0 * n_input)

    for i in range(1, n_filters):
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / n_input)

    return basis.T


def delta(feat, N):
    if N < 1:
        raise ValueError('N must be an integer >= 1')
    denominator = 2 * sum([i ** 2 for i in range(1, N + 1)])
    delta_rersult = reduce(lambda a, b: a + b,
                           [_delta_order(feat, i) for i in
                            range(1, 1 + N)])
    delta_rersult = delta_rersult / denominator

    return delta_rersult