#==========================================
# Header
#==========================================
# Copyright (c) Takuo Matsubara
# All rights reserved.
# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.



#==========================================
# Import Library
#==========================================
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import torch
from torch.func import jacrev

import sys
sys.path.insert(0, '../src')
from swgboost import SWGBoost
from model import Categorical
from datasets import dataset_loader
from learners import learner_loader



#==========================================
# Main Functions
#==========================================
def parse_arguments():
    argparser = ArgumentParser()
    argparser.add_argument("--id", type=str, default="03_uci")
    argparser.add_argument("--n_jobs", type=int, default=5)
    argparser.add_argument("--dataset", type=str, default="segment")
    argparser.add_argument("--learner", type=str, default="tree")
    argparser.add_argument("--learning_rate", type=float, default=0.4)
    argparser.add_argument("--n_estimators", type=int, default=4000)
    argparser.add_argument("--n_particles", type=int, default=10)
    argparser.add_argument("--bandwidth", type=float, default=0.1)
    argparser.add_argument("--subsample", type=float, default=1.0)
    return argparser.parse_args()


def preprocess_standardisation(dat_train, dat_test):
    if dat_train.shape[1] > 1:
        dat_train_mean, dat_train_std = np.mean(dat_train, axis=0), np.std(dat_train, axis=0)
        dat_train_std[dat_train_std == 0] = 1.0
    else:
        dat_train_mean, dat_train_std = np.mean(dat_train), np.std(dat_train)
    dat_train = ( dat_train - dat_train_mean ) / dat_train_std
    dat_test = ( dat_test - dat_train_mean ) / dat_train_std
    return (dat_train, dat_test), (dat_train_mean, dat_train_std)


def create_random_repeat(num_data, random_states):
    folds = []
    for random_state in random_states:
        np.random.seed(random_state)
        index = np.random.permutation(num_data)
        train_index = index[:int(0.8 * num_data)]
        test_index = index[int(0.8 * num_data):]
        folds.append((train_index, test_index))
    return folds


def extract_data(data, train_index, test_index):
    X, Y = data.values[:, 0:-1], data.values[:, -1]
    N_class = int(np.max(Y)) + 1

    def get_one_hot(ids_class, num_class):
        return np.eye(num_class)[ids_class.reshape(-1)]
    
    X_train = X[train_index]
    Y_train = get_one_hot(Y[train_index].astype('int'), N_class)
    X_test = X[test_index]
    Y_test = get_one_hot(Y[test_index].astype('int'), N_class)
    return X_train, Y_train, X_test, Y_test, N_class


def get_grad_funcs():
    scale = 10
    
    def log_posterior(p, y):
        return ( p - torch.log1p(p.exp().sum()) ) @ y[:-1] - torch.log1p(p.exp().sum()) * y[-1] - 0.5 * p @ p / (scale**2)
    
    grad_logp = jacrev(log_posterior, argnums=0)
    hess_logp_full = jacrev(jacrev(log_posterior, argnums=0), argnums=0)
    hess_logp = lambda p, y: torch.diagonal(hess_logp_full(p, y), 0)
    
    return grad_logp, hess_logp


def one_fold(data, data_ood, kth, fold_kth):
    train_index = fold_kth[0]
    test_index = fold_kth[1]
    X_train, Y_train, X_test, Y_test, N_class = extract_data(data, train_index, test_index)
    X_ood = data_ood.values[:,0:-1]

    model = Categorical()
    grad_logp, hess_logp = get_grad_funcs()

    reg = SWGBoost(grad_logp, hess_logp, learner_loader[args.learner]['class'],
        learner_param = learner_loader[args.learner]['default_param'],
        learning_rate = args.learning_rate,
        n_estimators = args.n_estimators,
        n_particles = args.n_particles,
        d_particles = N_class-1,
        bandwidth = args.bandwidth,
        subsample = args.subsample)
    reg.fit(X_train, Y_train)

    P_test = reg.predict(X_test)
    A_test = model.accuracy(Y_test, P_test)

    P_ood = reg.predict(X_ood)
    D_ood = model.ood_detection(P_test, P_ood)

    return np.array([ A_test, D_ood ])


def main(args):
    np.random.seed(1)
    torch.manual_seed(1)
    torch.set_num_threads(1)
    
    # load dataset
    data = dataset_loader[args.dataset]()
    data_ood = dataset_loader[args.dataset+"_ood"]()
    
    # 5 random seeds to shuffle data found from files in https://github.com/sharpenb/Posterior-Network
    folds = create_random_repeat(data.shape[0], [322, 365, 382, 510, 988])
    
    results_list = Parallel(n_jobs=args.n_jobs, backend="multiprocessing")(delayed(one_fold)(data, data_ood, kth, folds[kth]) for kth in range(len(folds)))
    results = np.array(results_list)
    
    print("=== Accuracy Test Summary {:.4f} +/- {:.4f} ===".format(np.mean(results[:,0]), np.std(results[:,0])))
    print("=== OOD Test Summary {:.4f} +/- {:.4f} ===".format(np.mean(results[:,1]), np.std(results[:,1])))
    pd.DataFrame(results).to_csv("../result/" + args.id + "_test_accuracy_ood_" + args.dataset + ".csv", index=False, header=["Accuracy", "OOD APS"])



#==========================================
# Execution
#==========================================
if __name__ == "__main__":
    args = parse_arguments()
    main(args)


