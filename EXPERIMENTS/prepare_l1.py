#!/usr/bin/env python3

from enum import Enum
import sys
import struct
import numpy as np
import h5py
import os.path
import click


class Scale(Enum):
    SMALL = 0
    MEDIUM = 1
    LARGE = 2


# DATASETS = {
#     Scale.SMALL: ['audio', 'mnist', 'enron'],
#     Scale.MEDIUM: ['glove', 'GIST1M', 'SIFT1M'],
#     Scale.LARGE: ['GIST80M', "SIFT1B"]
# }
DATASET_INFO = {}
CUR_PATH = os.path.dirname(__file__)


def get_dim(h5filename):
    name, _ = os.path.splitext(h5filename)
    return int(name.split('-')[-1])


# def uint64_to_binary(num):
#     return [float(c) for c in f"{num:0>64b}"]


def get_dsname(h5filename):
    dsname, _, _ = h5filename.partition('-')
    return dsname


def convert(h5filename, odir):
    assert os.path.exists(odir)
    print(f'Converting {h5filename} ...')
    _, name = os.path.split(h5filename)
    filename_prefix, _ = os.path.splitext(name)
    # dsname = get_dsname(filename_prefix)
    dim = get_dim(h5filename)
    qn = -1
    with h5py.File(h5filename, 'r') as inf:
        train_fn = f"{odir}/{filename_prefix}-train.fvecs"
        test_bfn = f"{odir}/{filename_prefix}-test.fvecs"
        train_tfn = f"{odir}/{filename_prefix}-train.txt"
        test_tfn = f"{odir}/{filename_prefix}-test.txt"

        with open(train_fn, 'wb') as df:
            with open(test_bfn, 'wb') as qf:
                with open(train_tfn, 'w') as dtf:
                    with open(test_tfn, 'w') as qtf:
                        train = inf['train'][:]

                        n = (int)(train.shape[0] / dim)
                        
                        print(f"#dim: {dim}, #points: {n}")

                        query = inf['test'][:]
                

                        qn = (int)(query.shape[0] / dim)
                        universe = max(np.max(train),np.max(query))


                        cnt = 0
                        for i in range(n):
                            df.write(struct.pack('i', dim))
                            point = train[(i * dim):((i+1) * dim)]
                            for val in point:
                                df.write(struct.pack('f', float(val)))
                                cnt += 1
                            
                            bs = []
                            for j in range(dim):
                                bs.append(str(int(train[i*dim+j])))
                            
                            sbs = " ".join(bs)
                            # it seems srs only accepts integer index
                            # oudf.write(f'{i + 1} {sbs}\n')
                            dtf.write(sbs + '\n')

                        # assert (cnt == n * dim)

                        cnt = 0
                        for i in range(qn):
                            qf.write(struct.pack('i', dim))
                            point = query[(i * dim):((i+1) * dim)]
                            for val in point:
                                qf.write(struct.pack('f', float(val)))
                                cnt += 1
                            
                            qbs = []
                            for j in range(dim):
                                qbs.append(str(int(query[i*dim+j])))
                            
                            sqbs = " ".join(qbs)
                            # it seems srs only accepts integer index
                            # oudf.write(f'{i + 1} {sbs}\n')
                            qtf.write(str(i) + ' ' + sqbs + '\n')

                        # assert (cnt == qn * dim)

                        return {
                            'train-b': os.path.split(train_fn)[-1],
                            'test-b': os.path.split(test_bfn)[-1],
                            'train-t':os.path.split(train_tfn)[-1],
                            'test-t':os.path.split(test_tfn)[-1],
                            'n': int(n),
                            'dimension': int(dim),
                            'qn': int(qn),
                            'dsh5': name,
                            'universe': int(universe)
                        }


def prepare_datasets():
    ddir = os.path.join(CUR_PATH, 'datasets')
    for f in os.listdir(ddir):
        if f.endswith('.h5'):
            dsname = get_dsname(f)
            if os.path.exists(dsname):
                print("directory already exists")
                exit(1)
            os.makedirs(dsname)
            DATASET_INFO[dsname] = convert(os.path.join(ddir, f), dsname)
    with open('datasets_info.json', 'w') as jf:
        import json
        json.dump(DATASET_INFO, jf, indent=4)
    with open('dataset_info.txt', 'w') as tf:
        # for ds, info in DATASET_INFO.items():
        for ds in sorted(DATASET_INFO.keys(), key=lambda k: (DATASET_INFO[k]['n'], DATASET_INFO[k]['dimension'])):
            info = DATASET_INFO[ds]
            trainb = info['train-b']
            testb = info['test-b']
            traint = info['train-t']
            testt = info['test-t']
            n = info['n']
            dim = info['dimension']
            qn = info['qn']
            dsh5 = info['dsh5']
            universe = info['universe']
            tf.write(f"{ds}%{trainb}%{testb}%{traint}%{testt}%{n}%{dim}%{qn}%{dsh5}%{universe}\n")


# CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

# @click.command(context_settings=CONTEXT_SETTINGS)
# @click.option('--no-large', default=True)
# def prepare(no_large):
#     """Prepare datasets."""
#     prepare_datasets()
#     if not no_large:
#         prepare_large_datasets()

if __name__ == '__main__':
    prepare_datasets()
