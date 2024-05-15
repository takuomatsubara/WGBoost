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
import torch
from torch.func import jacrev
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, '../src')
from fwgboost import FWGBoost
from swgboost import SWGBoost
from nwgboost import NWGBoost
from lgboost import LGBoost
from sklearn.tree import DecisionTreeRegressor



#==========================================
# Main Functions
#==========================================
def parse_arguments():
    argparser = ArgumentParser()
    argparser.add_argument("--id", type=str, default="supplement")
    argparser.add_argument("--max_estimators", type=int, default=100)
    argparser.add_argument("--max_depth", type=int, default=3)
    argparser.add_argument("--offset", type=int, default=10)
    return argparser.parse_args()


def plot_result(results_T, results_S, offset=1):
    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    
    sns.lineplot(x=np.log10(np.arange(offset,args.max_estimators+1)), y=np.log10(results_T[0,offset-1:]), linewidth=2, ax=ax)
    sns.lineplot(x=np.log10(np.arange(offset,args.max_estimators+1)), y=np.log10(results_T[1,offset-1:]), linewidth=2, ax=ax)
    sns.lineplot(x=np.log10(np.arange(offset,args.max_estimators+1)), y=np.log10(results_T[2,offset-1:]), linewidth=2, ax=ax)
    sns.lineplot(x=np.log10(np.arange(offset,args.max_estimators+1)), y=np.log10(results_T[3,offset-1:]), linewidth=2, ax=ax)
    
    ax.set_xlabel(r"common-log base learner number", fontsize=18)
    ax.set_ylabel(r"common-log computation time", fontsize=18)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    fig.tight_layout()
    fig.savefig("./result/"+args.id+"_time.png", dpi=288)

    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    
    sns.lineplot(x=np.arange(offset,args.max_estimators+1), y=results_S[0,offset-1:], linewidth=2, ax=ax, label="FKA-WGBoost")
    sns.lineplot(x=np.arange(offset,args.max_estimators+1), y=results_S[1,offset-1:], linewidth=2, ax=ax, label="SKA-WGBoost")
    sns.lineplot(x=np.arange(offset,args.max_estimators+1), y=results_S[2,offset-1:], linewidth=2, ax=ax, label="NKA-WGBoost")
    sns.lineplot(x=np.arange(offset,args.max_estimators+1), y=results_S[3,offset-1:], linewidth=2, ax=ax, label="LGBoost")
    
    ax.set_xlabel(r"base learner number", fontsize=18)
    ax.set_ylabel(r"maximum mean discrepancy", fontsize=18)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=16)
    
    fig.tight_layout()
    fig.savefig("./result/"+args.id+"_score.png", dpi=288)


def get_result(reg, X, Y, filename, scale=0.5):
    np.random.seed(1)
    torch.manual_seed(1)

    fit_times = reg.fit(X, Y)

    X_test = np.linspace(-3.5, 3.5, 500).reshape(-1,1)
    P_test = reg.predict_eachitr(X_test).reshape((reg.n_estimators, X_test.shape[0], reg.n_particles))

    bandwidth = 0.025
    K1_test = np.exp( - ( P_test[:,:,:,np.newaxis] - P_test[:,:,np.newaxis,:] )**2 / bandwidth )
    K2_test = np.exp( - 0.5 * ( P_test - np.sin(X_test)[np.newaxis,:] )**2 / ( scale**2 + bandwidth/2.0 ) ) * ( np.sqrt(bandwidth/2.0) / np.sqrt(scale**2 + bandwidth/2.0) )
    K3_test = ( np.sqrt(bandwidth/2.0) / np.sqrt(2.0*scale**2 + bandwidth/2.0) )
    S_test = K1_test.mean(axis=-1).mean(axis=-1) - 2 * K2_test.mean(axis=-1) + K3_test

    plot_output(X_test, P_test[-1], filename)

    return fit_times, S_test.mean(axis=-1)


def plot_output(X_test, P_test, filename, scale=0.5):
    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-2.7, 2.7)
    ax.set_xlabel(r"$x$", fontsize=14)
    ax.set_ylabel(r"$y$", fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    cil = np.sin(X_test).flatten() + 1.96 * scale
    ciu = np.sin(X_test).flatten() - 1.96 * scale
    ax.fill_between(X_test.flatten(), cil, ciu, color='b', alpha=0.1)
    
    for ith in range(P_test.shape[1]):
        sns.lineplot(x=X_test.flatten(), y=P_test[:,ith].flatten(), linewidth=2, color="red", ax=ax)

    fig.tight_layout()
    fig.savefig("./result/"+args.id+"_output_"+filename+".png", dpi=288)


def get_fwgboost(m, scale=0.5):
    grad_logp = lambda p, y: - (p - y) / scale**2
    
    reg = Time_FWGBoost(grad_logp, DecisionTreeRegressor,
        learner_param = {'max_depth': args.max_depth, 'random_state': 1},
        learning_rate = 0.1,
        n_estimators = m,
        n_particles = 10,
        d_particles = 1,
        bandwidth = 0.1,
        init_iter = 0,
        init_locs = np.linspace(-10, 10, 10).reshape(-1, 1))
    
    return reg


def get_swgboost(m, scale=0.5):
    grad_logp = lambda p, y: - (p - y) / scale**2
    hess_logp = lambda p, y: - torch.ones(1) / scale**2
    
    reg = Time_SWGBoost(grad_logp, hess_logp, DecisionTreeRegressor,
        learner_param = {'max_depth': args.max_depth, 'random_state': 1},
        learning_rate = 0.1,
        n_estimators = m,
        n_particles = 10,
        d_particles = 1,
        bandwidth = 0.1,
        init_iter = 0,
        init_locs = np.linspace(-10, 10, 10).reshape(-1, 1))
    
    return reg


def get_nwgboost(m, scale=0.5):
    grad_logp = lambda p, y: - (p - y) / scale**2
    hess_logp = lambda p, y: - torch.ones(1,1) / scale**2
    
    reg = Time_NWGBoost(grad_logp, hess_logp, DecisionTreeRegressor,
        learner_param = {'max_depth': args.max_depth, 'random_state': 1},
        learning_rate = 0.1,
        n_estimators = m,
        n_particles = 10,
        d_particles = 1,
        bandwidth = 0.1,
        init_iter = 0,
        init_locs = np.linspace(-10, 10, 10).reshape(-1, 1))
    
    return reg


def get_lgboost(m, scale=0.5):
    grad_logp = lambda p, y: - (p - y) / scale**2
    
    reg = Time_LGBoost(grad_logp, DecisionTreeRegressor,
        learner_param = {'max_depth': args.max_depth, 'random_state': 1},
        learning_rate = 0.1,
        n_estimators = m,
        n_particles = 10,
        d_particles = 1,
        init_iter = 0,
        init_locs = np.linspace(-10, 10, 10).reshape(-1, 1))
    
    return reg


def main(args):
    np.random.seed(1)
    torch.manual_seed(1)
    torch.set_num_threads(1)
    
    X = np.linspace(-3.5, 3.5, 200).reshape(-1,1)
    Z = np.sin(X)

    results_T = np.zeros((4, args.max_estimators))
    results_S = np.zeros((4, args.max_estimators))
    results_T[0], results_S[0] = get_result(get_fwgboost(args.max_estimators), X, Z, "FKA-WGBoost")
    results_T[1], results_S[1] = get_result(get_swgboost(args.max_estimators), X, Z, "SKA-WGBoost")
    results_T[2], results_S[2] = get_result(get_nwgboost(args.max_estimators), X, Z, "NKA-WGBoost")
    results_T[3], results_S[3] = get_result(get_lgboost(args.max_estimators), X, Z, "LGBoost")
    plot_result(results_T, results_S, offset=args.offset)



#==========================================
# Main Class
#==========================================

class Time_FWGBoost(FWGBoost):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.base0 = self.compute_init_base(Y)
        fit_times = np.zeros(self.n_estimators)

        for ith in range(self.n_estimators):
            head_time = time.time()

            L = self.learner_class(**self.learner_param)
            
            P = self.predict(X)
            G = self.gradient(P, Y)
            L.fit(X, self._reshape_backward(G))
            
            self.bases.append(L)
            self.rates.append(1.0)

            tail_time = time.time()
            fit_times[ith] = tail_time - head_time
        
        return np.cumsum(fit_times)
    

class Time_SWGBoost(SWGBoost):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.base0 = self.compute_init_base(Y)
        fit_times = np.zeros(self.n_estimators)

        for ith in range(self.n_estimators):
            head_time = time.time()

            L = self.learner_class(**self.learner_param)
            
            P = self.predict(X)
            G = self.gradient(P, Y)
            L.fit(X, self._reshape_backward(G))
            
            self.bases.append(L)
            self.rates.append(1.0)

            tail_time = time.time()
            fit_times[ith] = tail_time - head_time
            
        return np.cumsum(fit_times)
    

class Time_NWGBoost(NWGBoost):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.base0 = self.compute_init_base(Y)
        fit_times = np.zeros(self.n_estimators)

        for ith in range(self.n_estimators):
            head_time = time.time()

            L = self.learner_class(**self.learner_param)
            
            P = self.predict(X)
            G = self.gradient(P, Y)
            L.fit(X, self._reshape_backward(G))
            
            self.bases.append(L)
            self.rates.append(1.0)

            tail_time = time.time()
            fit_times[ith] = tail_time - head_time
            
        return np.cumsum(fit_times)
    

class Time_LGBoost(LGBoost):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.base0 = self.compute_init_base(Y)
        fit_times = np.zeros(self.n_estimators)

        for ith in range(self.n_estimators):
            head_time = time.time()

            L = self.learner_class(**self.learner_param)
            
            P = self.predict(X)
            G = self.gradient(P, Y)
            L.fit(X, self._reshape_backward(G))
            
            self.bases.append(L)
            self.rates.append(1.0)

            tail_time = time.time()
            fit_times[ith] = tail_time - head_time
        
        return np.cumsum(fit_times)



#==========================================
# Execution
#==========================================
if __name__ == "__main__":
    args = parse_arguments()
    main(args)


