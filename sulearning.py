from __future__ import division
import argparse
import time
import numpy as np
import heapq

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn import svm
from sklearn import random_projection
import os
from nsu_data_generation import load_dataset, convert_su_data_sklearn_compatible
from mpe import wrapper

import sys
import re

seed = 0
class SU_Base(BaseEstimator, ClassifierMixin):

    def __init__(self, prior=.7, noise=.2, su_prior=.7, lam=1):
        self.prior = prior
        self.noise = noise
        self.su_prior = su_prior
        self.lam = lam

    def fit(self, x, y):
        pass

    def predict(self, x):
        check_is_fitted(self, 'coef_')
        x = check_array(x)
        x = self._basis(x)
        return np.sign(.1 + np.sign(x.dot(self.coef_)))


    def score(self, x, y):
        x_s, x_u = x[y == 1, :], x[y == 0, :]
        f = self.predict
        p_p = self.su_prior
        p_n = 1 - self.su_prior
        p_s = p_p ** 2 + p_n ** 2

        # SU risk estimator with zero-one loss
        r_s = (np.sign(-f(x_s)) - np.sign(f(x_s))) * p_s / (p_p - p_n)
        r_u = (-p_n * (1 - np.sign(f(x_u))) + p_p * (1 - np.sign(-f(x_u)))) / (p_p - p_n)
        risk = r_s.mean() + r_u.mean()

        # makes higher score means good performance
        score = np.maximum(0, 1 - risk)
        return score


    def _basis(self, x):
        # linear basis
        return np.hstack((x, np.ones((len(x), 1))))


class SU_SL(SU_Base):

    def fit(self, x, y):
        check_classification_targets(y)
        x, y = check_X_y(x, y)
        x_s, x_u = x[y == +1, :], x[y == 0, :]
        n_s, n_u = len(x_s), len(x_u)

        p_p = self.prior
        p_n = 1 - self.prior
        p_s = p_p ** 2 + p_n ** 2
        k_s = self._basis(x_s)
        k_u = self._basis(x_u)
        d = k_u.shape[1]
        r_d = self.noise

        su_p_p = self.su_prior
        su_p_n = 1 - su_p_p
        su_p_s = su_p_p ** 2 + su_p_n ** 2


        A = (p_s * (1 - p_s)) / ((1 - r_d - p_s) * (p_p - p_n))
        B = (p_s * r_d - p_n * (r_d + p_s - 1)) / ((r_d + p_s - 1) * (p_p - p_n))
        C = (p_s * r_d - p_p * (r_d + p_s - 1)) / ((r_d + p_s - 1) * (p_p - p_n))

        D = (k_u.T.dot(k_u) + 2 * self.lam * n_u * np.eye(d)) * 1 / n_u
        h = 2 * A * k_s.T.mean(axis=1) + (B + C) * k_u.T.mean(axis=1)
        self.coef_ = np.linalg.solve(D, h)


        M = (su_p_p - su_p_n) / n_u * (k_u.T.dot(k_u) + 2 * self.lam * n_u * np.eye(d))
        e = 2 * su_p_s * k_s.T.mean(axis=1) - k_u.T.mean(axis=1)
        global cofeSLSU
        cofeSLSU = np.linalg.solve(M, e)

        return self



def class_prior_estimation(DS, DU):
    # class-prior estimation using MPE method in Ramaswamy et al. (2016)
    from mpe import wrapper
    print('1:')
    mpe_time_start = time.time()
    km1, km2 = wrapper(DU, DS, 0.88)
    gamma = km2
    su_p = gamma
    print('gamma:{}'.format(gamma))
    print('2:')
    km1, km2 = wrapper(DS, DU, 2-1/gamma)
    mpe_time_end = time.time()
    kappa = km2
    print('kappa:{}'.format(kappa))
    pi_s = gamma * (1 - kappa) / (1 - gamma * kappa)
    rho_d = kappa * (1 - gamma) / (1 - gamma * kappa)
    print('MPE Time cost:', mpe_time_end - mpe_time_start, 's')
    return pi_s, rho_d, su_p

def predict_SU(x, cofe):
    x = np.hstack((x, np.ones((len(x), 1))))
    return np.sign(.1 + np.sign(x.dot(cofe)))

def averagenum(num):
    nsum = 0
    if len(num) == 0 :
        return 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)


def main(dataset, mpe_num=400, prior=0.7, n_s=500, n_u=500, noise=0.2, seed = 1, end_to_end=False):
    SU = SU_SL

    print('prior =', prior, '|noise =', noise, '|n_s =', n_s, '|n_u =', n_u, '|seed =', seed,
          '|mpe num =', mpe_num)
    print('Dataset:', dataset)
    print('---------------------------------------')

    n_test = 5000

    iteration_time_start = time.time()
    # print seed, 'th seed'

    x_s, x_u, x_test, y_test, mpe_xs, mpe_xu = \
        load_dataset(mpe_num, n_s, n_u, n_test, prior, noise, dataset, seed)
    x_train, y_train = convert_su_data_sklearn_compatible(x_s, x_u)
    print(mpe_xs.shape, mpe_xu.shape)



    if end_to_end:
        # use KM2 (Ramaswamy et al., 2016)

        print('MPE:')
        est_pi_s, est_noise, su_est_prior = class_prior_estimation(mpe_xs, mpe_xu)
        print('su_est_prior:', su_est_prior)
        print('est_pi_s:', est_pi_s)
        print('est_noise:', est_noise)
        if est_pi_s > 0.5 > est_noise:
            est_prior = 0.5 * (np.sqrt(2 * est_pi_s - 1) + 1)
            print('Using MPE estimation')
        else:
            exit('MPE fault')



    else:
        su_est_prior = prior
        est_prior = prior
        est_noise = noise
        print('Using true values')


    print('est_prior:', est_prior, 'est_noise:', est_noise, 'su_est_prior:', su_est_prior)

    # training with the best hyperparameter
    lam_best = 1e-04



    clf = SU(prior=est_prior, noise=est_noise, su_prior=su_est_prior, lam=lam_best)
    clf.fit(x_train, y_train)
    # test prediction
    y_nsu_pred = clf.predict(x_test)
    y_su_pred = predict_SU(x_test, cofeSLSU)
    sl_nsu_acc = accuracy_score(y_test, y_nsu_pred)
    sl_su_acc = accuracy_score(y_test, y_su_pred)

    print('SL N_SU acc:%f' % (sl_nsu_acc))
    print('SL Baseline acc:%f' % (sl_su_acc))
    iteration_time_end = time.time()
    print('Seed Time cost:', iteration_time_end - iteration_time_start, 's')
    print('----------------------')

    return sl_nsu_acc, sl_su_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',
        action   = 'store',
        required = True,
        type     = str,
        help     = 'dataset')

    parser.add_argument('--mpe',
        action   = 'store',
        required = False,
        type     = int,
        default  = 400,
        help     = 'number of MPE data')

    parser.add_argument('--ns',
        action   = 'store',
        required = False,
        type     = int,
        default  = 500,
        help     = 'number of similar data pairs')

    parser.add_argument('--nu',
        action   = 'store',
        required = False,
        type     = int,
        default  = 500,
        help     = 'number of unlabeled data points')

    parser.add_argument('--prior',
        action   = 'store',
        required = False,
        type     = float,
        default  = 0.7,
        help     = 'true class-prior (ratio of positive data)')

    parser.add_argument('--noise',
        action   = 'store',
        required = False,
        type     = float,
        default  = 0.2,
        help     = 'noise rate (ratio of d-pairs in s-data)')

    parser.add_argument('--seed',
        action   = 'store',
        required = False,
        type     = int,
        default  = 1,
        help     = 'random seed')


    parser.add_argument('--full',
        action   = 'store_true',
        default  = False,
        help     = 'do end-to-end experiment including class-prior estimation (default: false)')

    parser.add_argument('--p',
                        action='store',
                        required=False,
                        type=int,
                        default=0,
                        help='1:print in terminal; 0:print in txt file')
    parser.add_argument('--gpu', type=str, default='n')

    args = parser.parse_args()

    outdir = './result/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if args.p == 0:
        f = open(outdir + args.dataset + '_' + str(args.prior) + '_' + str(args.noise) + '.txt', 'a')
        sys.stdout = f
        sys.stderr = f

    sl_su_acc_list = []
    sl_nsu_acc_list = []
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for i in range(args.seed):
        seed = seed + 1
        nsu_acc, su_acc = main(args.dataset, args.mpe, args.prior, args.ns, args.nu, args.noise, seed, args.full)
        nsu_acc = max(nsu_acc, 1-nsu_acc)
        su_acc = max(su_acc, 1 - su_acc)
        sl_nsu_acc_list.append(nsu_acc)
        sl_su_acc_list.append(su_acc)
    print('SL N_SU acc list:', sl_nsu_acc_list)
    print('SL N_SU acc average:', end=' ')
    print(round((np.array(sl_nsu_acc_list).mean()), 3))
    print('SL N_SU acc std:', end=' ')
    print(round(np.std(np.array(sl_nsu_acc_list), ddof=1), 3))
    print(round((100 * np.array(sl_nsu_acc_list).mean()), 1), '$\pm$',
          round(100 * np.std(np.array(sl_nsu_acc_list), ddof=1), 1))
    print('SL Baseline acc list:', sl_su_acc_list)
    print('SL Baseline acc average:', end=' ')
    print(round((np.array(sl_su_acc_list).mean()), 3))
    print('SL Baseline acc std:', end=' ')
    print(round(np.std(np.array(sl_su_acc_list), ddof=1), 3))
    print(round((100 * np.array(sl_su_acc_list).mean()), 1), '$\pm$',
          round(100 * np.std(np.array(sl_su_acc_list), ddof=1), 1))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

