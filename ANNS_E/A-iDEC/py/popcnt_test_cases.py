#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
from jinja2 import Environment, FileSystemLoader
import os.path
import numpy as np


# uniform
def gen_popcnt_test_cases(n_bits, n_samples):
    """
    """
    assert n_bits <= 64

    test_in = []
    test_out = []
    if n_bits < 14:
        for x in range(2 ** n_bits):
            x_str = '{0:0{w}b}'.format(x, w=n_bits)
            x_arr = np.array([int(s) for s in x_str])
            test_in.append('0b' + x_str)
            test_out.append(np.sum(x_arr))
    else:
        bits = np.random.randint(0, high=2, size=(n_samples, n_bits))
        for xb in bits:
            test_in.append('0b' + ''.join([str(b) for b in xb]))
            test_out.append(np.sum(xb))
    return test_in, test_out


# uniform
def gen_popcnt_test_cases_vr(n_bits, n_samples, rate=None):
    """
    """
    assert n_bits <= 64
    if (n_bits < 14) or (rate is None) or (rate == 0.5):
        return gen_popcnt_test_cases(n_bits, n_samples)
    # bits = np.random.randint(0, high=2, size=(n_samples, n_bits))
    rv = np.random.rand(n_samples, n_bits)
    bits = np.zeros((n_samples, n_bits), dtype=int)
    bits[rv < rate] = 1
    test_in = []
    test_out = []
    for xb in bits:
        test_in.append('0b' + ''.join([str(b) for b in xb]))
        test_out.append(np.sum(xb))
    return test_in, test_out


def gen_vector(dim, dtype):
    if dtype == 'float':
        return np.array(np.random.randn(dim), dtype=np.float32)
    elif dtype == 'int':
        return np.random.randint(0, high=256, size=dim)
    elif dtype == 'bit' or dtype == 'bit_enc':
        return np.random.randint(0, high=2, size=dim)


def stringtify(vec, dtype, word_size=None):
    """

    :param vec:
    :param dtype:
    :return:
    """
    if (dtype is not 'bit') and (dtype is not 'bit_enc'):
        s = ', '.join(['{}'.format(vi) for vi in vec])
    else:
        dim = len(vec)
        tmp = []
        if word_size is None:
            word_size = 64
        i = 0
        while dim >= word_size:
            tmp.append(int(''.join([str(vi) for vi in vec[i:(i + word_size)]]), 2))
            dim -= word_size
            i += word_size
        if dim > 0:
            tmp.append(int(''.join([str(vi) for vi in vec[i:]]), 2))

        suf = 'u'
        s = ','.join([str(ti) + suf for ti in tmp])
    return s


def gen_dot_prod_test_cases(dim, n_samples, dtype1, dtype2, rtype, word_size=None):
    """

    :param dim:
    :param n_samples:
    :param dtype1:
    :param dtype2:
    :return:
    """
    tests_ = []
    for _ in range(n_samples):
        va = np.array(gen_vector(dim, dtype=dtype1))
        vb = np.array(gen_vector(dim, dtype=dtype2))

        tests_.append(
            {
                'va': {
                    'data': stringtify(va, dtype1, word_size),
                    'dtype': dtype1 if (dtype1 != 'bit' and dtype1 != 'bit_enc') else 'uint{}_t'.format(word_size)
                },
                'vb': {
                    'data': stringtify(vb, dtype2, word_size),
                    'dtype': dtype2 if (dtype2 != 'bit' and dtype2 != 'bit_enc') else 'uint{}_t'.format(word_size)
                },
                'dlen': dim,
                'res': 0,
                'rtype': rtype
            }
        )
        if dtype1 == 'bit_enc':
            va[va == 0] = -1
        if dtype2 == 'bit_enc':
            vb[vb == 0] = -1
        tests_[-1]['res'] = np.dot(va, vb)
    return tests_


# Capture our current directory
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_dot_prod_cpp_test_file(template_filename, test_cases, n_tests_each=200, out_filename=None):
    # Create the jinja2 environment.
    # Notice the use of trim_blocks, which greatly helps control whitespace.
    j2_env = Environment(loader=FileSystemLoader(os.path.join(THIS_DIR, 'templates')),
                         trim_blocks=True)
    if out_filename is None:
        filename, file_extension = os.path.splitext(template_filename)
        out_filename = filename
    test_dims = (2, 4, 8, 16, 32)
    test_dims_compact = (64, 128, 256, 60, 59, 100, 119, 24, 23, 150)
    config = []
    type_name = {
        'int': 'int /* {} */',
        'float': 'float /* {} */',
        'bit': 'uint{}_t',
        'bit_enc': 'uint{}_t'
    }
    for test_case in test_cases:
        if test_case[1].find('cb') != -1:
            tdims = test_dims_compact
        else:
            tdims = test_dims
        func_suf = {
            'dot_prod': '<{t1}, {t2}, {res}>',
            'dot_prod_cb': '<{t1}, {res}, {t2}>',
            'dot_prod_cb_enc': '<{t1}, {res}, {t2}>',
            'dot_prod_cb_cb_enc': '<{res}, {t}>'
        }
        for d in tdims:
            ws = test_case[-1]
            if isinstance(ws, str):
                ws = None
            tests = gen_dot_prod_test_cases(d, n_tests_each, test_case[2], test_case[3], test_case[4], word_size=ws)
            ws_str = ws if ws is not None else ''
            config.append({
                'description': test_case[0].format(d),
                'tests': tests,
                'func': test_case[1] + func_suf[test_case[1]].format(t1=type_name[test_case[2]].format(ws_str),
                                                                     t2=type_name[test_case[3]].format(ws_str),
                                                                     res=type_name[test_case[4]].format(ws_str)) if
                test_case[
                    1] != 'dot_prod_cb_cb_enc' else
                test_case[1] + func_suf[test_case[1]].format(t=type_name[test_case[2]].format(ws_str),
                                                             res=type_name[test_case[4]].format(ws_str))
            })
    with open(out_filename, 'w') as wfp:
        wfp.write(j2_env.get_template(template_filename).render(all_tests=config))


def generate_cc_test_file(template_filename, n_tests_each=200, out_filename=None):
    # Create the jinja2 environment.
    # Notice the use of trim_blocks, which greatly helps control whitespace.
    j2_env = Environment(loader=FileSystemLoader(os.path.join(THIS_DIR, 'templates')),
                         trim_blocks=True)
    if out_filename is None:
        filename, file_extension = os.path.splitext(template_filename)
        out_filename = filename
    test_cases = (
        (32, 'unsigned_int_tests'), (64, 'unsigned_long_long_tests'),
        (8, 'uint8_t_tests'), (16, 'uint16_t_tests'), (32, 'uint32_t_tests'),
        (64, 'uint64_t_tests'), (16, 'unsigned_short_tests'), (32, 'unsigned_long_tests')
    )
    config = {}
    #
    rate = None
    for nb, key in test_cases:
        k = n_tests_each
        test_in = []
        test_out = []
        for rate in (0.2, 0.4, 0.6, 0.8):
            test_in_, test_out_ = gen_popcnt_test_cases_vr(nb, n_samples=k, rate=rate)
            test_in += test_in_
            test_out += test_out_
        test_in.append('0b0')
        test_out.append(0)
        test_in.append('0b' + ('1' * nb))
        test_out.append(nb)
        config[key] = [
            {'in': i, 'out': o} for i, o in zip(test_in, test_out)
        ]
    with open(out_filename, 'w') as wfp:
        wfp.write(j2_env.get_template(template_filename).render(tests=config))


if __name__ == '__main__':
    generate_cc_test_file("popcnt_test.cpp.template", n_tests_each=10, out_filename="popcnt_s_test.cpp")
    generate_cc_test_file("popcnt_test.cpp.template", n_tests_each=200, out_filename="popcnt_l_test.cpp")
    for test_cases in (
            (
            ('T dot_prod(const T *,const T *, size_t) with T=int, dim={}', 'dot_prod', 'int', 'int', 'int'),
            ('T dot_prod(const T *,const T *, size_t) with T=float, dim={}', 'dot_prod', 'float', 'float',
             'float'),
            ('float dot_prod(const T1 *,const T2 *, size_t) with T1=int, T2=float, dim={}', 'dot_prod', 'int',
             'float',
             'float'),
            ('float dot_prod(const T1 *,const T2 *, size_t) with T1=float, T2=int, dim={}', 'dot_prod', 'float',
             'int',
             'float')),
            (('int dot_prod_cb(const int *,const uint32_t *, size_t), dim={}', 'dot_prod_cb', 'int', 'bit', 'int', 32),
             ('float dot_prod_cb(const float *,const uint32_t *, size_t), dim={}', 'dot_prod_cb', 'float', 'bit',
              'float',
              32),
             ('int dot_prod_cb(const int*, const uint64_t*, size_t), dim={}', 'dot_prod_cb', 'int', 'bit', 'int', 64),
             (
                     'float dot_prod_cb(const float*, const uint64_t*, size_t), dim={}', 'dot_prod_cb', 'float', 'bit',
                     'float',
                     64)),
            (('int dot_prod_cb_enc(const int*, const uint32_t*, size_t), dim={}', 'dot_prod_cb_enc', 'int', 'bit_enc',
              'int',
              32),
             ('float dot_prod_cb_enc(const float*, const uint32_t*, size_t), dim={}', 'dot_prod_cb_enc', 'float',
              'bit_enc',
              'float', 32),
             (
                     'int dot_prod_cb_enc(const int*, const uint64_t*, size_t), dim={}', 'dot_prod_cb_enc', 'int',
                     'bit_enc', 'int',
                     64),
             ('float dot_prod_cb_enc(const float*, const uint64_t*, size_t), dim={}', 'dot_prod_cb_enc', 'float',
              'bit_enc',
              'float', 64)),
            ((
                     'int dot_prod_cb_cb_enc(const uint64_t*, const uint64_t*, size_t), dim={}', 'dot_prod_cb_cb_enc',
                     'bit_enc',
                     'bit_enc',
                     'int', 64),
             ('int dot_prod_cb_cb_enc(const uint32_t*, const uint32_t*, size_t), , dim={}', 'dot_prod_cb_cb_enc',
              'bit_enc',
              'bit_enc', 'int', 32))):
        generate_dot_prod_cpp_test_file('dot_prod_test.cpp.template',
                                        test_cases=test_cases,
                                        n_tests_each=5,
                                        out_filename=test_cases[0][1] + '_s_test.cpp')
        generate_dot_prod_cpp_test_file('dot_prod_test.cpp.template',
                                        test_cases=test_cases,
                                        n_tests_each=100,
                                        out_filename=test_cases[0][1] + '_l_test.cpp')
