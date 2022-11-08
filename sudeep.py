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
from model import NsuNet
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, normal_ as xavier, normal

import sys
import re
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

import matplotlib.pyplot as plt

def class_prior_estimation(DS, DU):
    # class-prior estimation using MPE method in Ramaswamy et al. (2016)
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

def averagenum(num):
    nsum = 0
    if len(num) == 0 :
        return 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)

def lg_loss(a,b):
    c = torch.exp(- a * b) + 1
    l = torch.log(c)
    return torch.mean(l)

def nsufit(epochs, nsu_model,  x, y, x_test, y_test, optimizer, scheduler, prior=.7, noise=.2):

    x_s, x_u = x[y == +1, :], x[y == 0, :]
    x_s = torch.from_numpy(x_s).float().cuda()
    x_u = torch.from_numpy(x_u).float().cuda()
    x_test = torch.from_numpy(x_test).float().cuda()

    p_p = prior
    p_n = 1 - prior
    p_s = p_p ** 2 + p_n ** 2
    r_d = noise

    A = (p_s * (1 - p_s)) / ((1 - r_d - p_s) * (p_p - p_n))
    B = (p_s * r_d - p_n * (r_d + p_s - 1)) / ((r_d + p_s - 1) * (p_p - p_n))
    C = (p_s * r_d - p_p * (r_d + p_s - 1)) / ((r_d + p_s - 1) * (p_p - p_n))

    nsu_loss_plt = []
    nsu_acc_plt = []


    for epoch in range(epochs):
        nsu_model.train()
        out_s = nsu_model(x_s)
        out_u = nsu_model(x_u)

        ones = torch.ones(len(out_s)).cuda()
        zeros = -torch.ones(len(out_s)).cuda()
        loss_s1 = lg_loss(out_s, ones)
        loss_s0 = lg_loss(out_s, zeros)
        loss_s = A * (loss_s1 - loss_s0)

        oneu = torch.ones(len(out_u)).cuda()
        zerou = -torch.ones(len(out_u)).cuda()
        loss_u1 = lg_loss(out_u, oneu)
        loss_u0 = lg_loss(out_u, zerou)
        loss_u = B * loss_u1 - C * loss_u0

        loss = loss_s + loss_u
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss.detach().cpu()
        nsu_model.eval()
        out = nsu_model(x_test).squeeze()
        out = out.detach().cpu()
        out = np.sign(.1 + np.sign(out))
        nsu_acc = accuracy_score(y_test, out)
        nsu_acc = max(nsu_acc, 1 - nsu_acc)
        nsu_loss_plt.append(loss.item())
        nsu_acc_plt.append(nsu_acc)

    index = nsu_loss_plt.index(min(nsu_loss_plt))
    print('NSU loss_min No.', index, 'acc:', nsu_acc_plt[index])
    val_acc = nsu_acc_plt[index]

    return val_acc


def sufit(epochs, su_model,  x, y, x_test, y_test, suoptimizer, suscheduler, su_prior=.7):
    x_s, x_u = x[y == +1, :], x[y == 0, :]
    x_s = torch.from_numpy(x_s).float().cuda()
    x_u = torch.from_numpy(x_u).float().cuda()
    x_test = torch.from_numpy(x_test).float().cuda()

    su_p_p = su_prior
    su_p_n = 1 - su_p_p
    su_p_s = su_p_p ** 2 + su_p_n ** 2

    D = su_p_s / (su_p_p - su_p_n)
    E = - su_p_n / (su_p_p - su_p_n)
    F = su_p_p / (su_p_p - su_p_n)

    su_loss_plt = []
    su_acc_plt = []

    for epoch in range(epochs):
        su_model.train()
        out_s = su_model(x_s).squeeze()
        out_u = su_model(x_u).squeeze()

        ones = torch.ones(len(out_s)).cuda()
        zeros = -torch.ones(len(out_s)).cuda()
        loss_s1 = lg_loss(out_s, ones)
        loss_s0 = lg_loss(out_s, zeros)
        loss_s = D * (loss_s1 - loss_s0)

        oneu = torch.ones(len(out_u)).cuda()
        zerou = -torch.ones(len(out_u)).cuda()
        loss_u1 = lg_loss(out_u, oneu)
        loss_u0 = lg_loss(out_u, zerou)
        loss_u = E * loss_u1 + F * loss_u0

        loss = loss_s + loss_u
        suoptimizer.zero_grad()
        loss.backward()
        suoptimizer.step()
        suscheduler.step()
        loss.detach().cpu()
        su_model.eval()
        out = su_model(x_test).squeeze()
        out = out.detach().cpu()
        out = np.sign(.1 + np.sign(out))

        su_acc = accuracy_score(y_test, out)
        su_acc = max(su_acc, 1 - su_acc)
        su_loss_plt.append(loss.item())
        su_acc_plt.append(su_acc)


    index = su_loss_plt.index(min(su_loss_plt))
    val_acc = su_acc_plt[index]


    return val_acc



def main(dataset, mpe_num=400, prior=0.7, n_s=500, n_u=500, noise=0.2, seed = 1, end_to_end=False):

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
    print('MPE data shape:', mpe_xs.shape, mpe_xu.shape)

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




    learning_rate = 0.002
    nsu_model = NsuNet(input_size=x_train.shape[1]).cuda()
    su_model = NsuNet(input_size=x_train.shape[1]).cuda()

    optimizer = torch.optim.SGD(nsu_model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    suoptimizer = torch.optim.SGD(su_model.parameters(), lr=learning_rate, momentum=0.9)
    suscheduler = torch.optim.lr_scheduler.StepLR(suoptimizer, step_size=40, gamma=0.1)
    args.epochs = 250
    nsu_val_acc = nsufit(args.epochs, nsu_model, x_train, y_train, x_test, y_test, optimizer, scheduler, prior=est_prior, noise=est_noise)
    su_val_acc = sufit(args.epochs, su_model, x_train, y_train, x_test, y_test, suoptimizer, suscheduler, su_prior=su_est_prior)


    iteration_time_end = time.time()
    print('Seed Time cost:', iteration_time_end - iteration_time_start, 's')
    print('----------------------')

    return nsu_val_acc, su_val_acc


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
                        action='store',
                        required=False,
                        type=int,
                        default=1,
                        help='random seed')

    parser.add_argument('--full',
                        action='store_true',
                        default=False,
                        help='do end-to-end experiment including class-prior estimation (default: false)')

    parser.add_argument('--p',
                        action='store',
                        required=False,
                        type=int,
                        default=0,
                        help='1:print in terminal; 0:print in txt file')
    parser.add_argument('--gpu', type=str, default='n')

    args = parser.parse_args()

    if args.gpu != 'n':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    outdir = './deep_cv/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if args.p == 0:
        f = open(outdir + args.dataset + '_' + str(args.prior) + '_' + str(args.noise) + '.txt', 'a')
        sys.stdout = f
        sys.stderr = f

    max_su_acc_list = []
    max_nsu_acc_list = []
    val_su_acc_list = []
    val_nsu_acc_list = []
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    for i in range(args.seed):
        seed = i + 1
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        nsu_val_acc, su_val_acc = main(args.dataset, args.mpe, args.prior, args.ns, args.nu, args.noise, seed, args.full)

        val_nsu_acc_list.append(nsu_val_acc)
        val_su_acc_list.append(su_val_acc)

    print('----------------------')
    print('N_SU val acc list:', val_nsu_acc_list)
    print(round((100 * np.array(val_nsu_acc_list).mean()), 1), '$\pm$',
          round(100 * np.std(np.array(val_nsu_acc_list), ddof=1), 1))
    print('Baseline val acc list:', val_su_acc_list)
    print(round((100 * np.array(val_su_acc_list).mean()), 1), '$\pm$', round(100 * np.std(np.array(val_su_acc_list), ddof=1), 1))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
