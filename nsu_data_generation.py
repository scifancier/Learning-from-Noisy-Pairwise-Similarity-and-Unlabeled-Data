#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas
import random


def pair_data_generator(data_p, data_n, nssp, nssn, nsd):
    index = np.random.choice(data_p.shape[0] ** 2, nssp, replace=False)
    row_index = index // data_p.shape[0]
    col_index = index - index // data_p.shape[0] * data_p.shape[0]
    xsp = np.hstack((data_p[row_index, 1:], data_p[col_index, 1:]))

    index = np.random.choice(data_n.shape[0] ** 2, nssn, replace=False)
    row_index = index // data_n.shape[0]
    col_index = index - index // data_n.shape[0] * data_n.shape[0]
    xsn = np.hstack((data_n[row_index, 1:], data_n[col_index, 1:]))

    index = np.random.choice(data_p.shape[0] * data_n.shape[0], nsd, replace=False)
    row_index = index // data_n.shape[0]
    col_index = index - index // data_n.shape[0] * data_n.shape[0]
    xd = np.hstack((data_p[row_index, 1:], data_n[col_index, 1:]))

    xs = np.concatenate((xsp, xsn, xd))

    return xs


def train_dataset(ns, nu, prior, noise, data_p, data_n, seed):
    nss = int(ns * (1 - noise))
    nsd = ns - nss
    nssp = int(nss * prior ** 2 / (prior ** 2 + (1 - prior) ** 2))
    nssn = nss - nssp

    nusp = int(nu * prior ** 2)
    nusn = int(nu * (1 - prior) ** 2)
    nud = nu - nusp - nusn

    np.random.seed(seed)

    xs = pair_data_generator(data_p, data_n, nssp, nssn, nsd)
    xu = pair_data_generator(data_p, data_n, nusp, nusn, nud)

    return xs, xu


def test_dataset(n, prior, data_p, data_n):
    xp = data_p[:, 1:]
    xn = data_n[:, 1:]
    yp = data_p[:, 0]
    yn = data_n[:, 0]
    x = np.concatenate((xp, xn))
    y = np.concatenate((yp, yn))
    y = y.reshape(-1)

    return x, y


def load_dataset(mpe_num, n_s, n_u, n_test, prior, noise, dataset, seed):
    tr_p = np.load("./data/" + dataset + "_p.npy")
    tr_n = np.load("./data/" + dataset + "_n.npy")
    np.random.seed(seed)
    np.random.shuffle(tr_p)
    np.random.shuffle(tr_n)
    te_p = tr_p[-tr_p.shape[0]//10:]
    tr_p = tr_p[:-tr_p.shape[0]//10]
    te_n = tr_n[-tr_n.shape[0]//10:]
    tr_n = tr_n[:-tr_n.shape[0]//10]

    # print tr_p.shape
    if tr_p.shape[0] > 2000:
        print('having p')
        tr_p = tr_p[np.random.choice(tr_p.shape[0], 2000, replace=False)]
    if tr_n.shape[0] > 2000:
        print('having n')
        tr_n = tr_n[np.random.choice(tr_n.shape[0], 1500, replace=False)]
    print('----------Generating', seed, ' Data-----------')
    x_s, x_u = train_dataset(n_s, n_u, prior, noise, tr_p, tr_n, seed)
    mpe_xs, mpe_xu = x_s, x_u
    x_test, y_test = test_dataset(n_test, prior, te_p, te_n)
    return x_s, x_u, x_test, y_test, mpe_xs, mpe_xu


def convert_su_data_sklearn_compatible(x_s, x_u):
    x = np.concatenate((x_s.reshape(-1, x_s.shape[1] // 2), x_u.reshape(-1, x_u.shape[1] // 2)))
    y = np.concatenate((np.ones(x_s.shape[0] * 2), np.zeros(x_u.shape[0] * 2)))
    return x, y