#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import h5py
from numpy import linalg as LA
from tqdm import tqdm
from tqdm import trange


def random_data(n, dim, max_range, p=2):
    """

    :param n:
    :param dim:
    :param max_range:
    :return:
    """
    raw_data_a = np.random.randint(0, high=max_range, size=(n, dim))
    raw_data_b = np.random.randint(0, high=max_range, size=(n, dim))
    distance = np.zeros(n)
    for i in trange(n):
        distance[i] = LA.norm(raw_data_a[i, :] - raw_data_b[i, :], ord=p)
    return raw_data_a, raw_data_b, distance


if __name__ == '__main__':
    n = 10000
    dim = 256
    max_range = 256

    for p in (1, 2, 3, 11):
        x, y, distance = random_data(n, dim, max_range, p)
        f = h5py.File('random_integer_data_p{}.hdf5'.format(p), 'w')
        f.create_dataset('n', (1,), data=n)
        f.create_dataset('dim', (1,), data=dim)
        f.create_dataset("X", (n, dim), data=x, dtype='i')
        f.create_dataset("Y", (n, dim), data=y, dtype='i')
        f.create_dataset("distance", (1, n), data=distance)
        f.close()
    
