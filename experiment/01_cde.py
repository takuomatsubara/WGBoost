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
import scipy.stats as stats
import pandas as pd
import torch
from torch.func import jacrev
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, '../src')
from swgboost import SWGBoost
from sklearn.tree import DecisionTreeRegressor



#==========================================
# Main Functions
#==========================================
def get_grad_funcs():
    alpha = 0.01
    beta = 0.01
    scale = 10

    def log_posterior(p0, p1, y):
        return - (1/2) * ( (y - p0)**2 ) / torch.exp(2*p1) - (1/2) * p0**2 / (scale**2) - ( alpha + 1 ) * p1 - beta / torch.exp(p1)
    
    log_grad_p0 = jacrev(log_posterior, argnums=0)
    log_grad_p1 = jacrev(log_posterior, argnums=1)
    log_hess_p0 = jacrev(jacrev(log_posterior, argnums=0), argnums=0)
    log_hess_p1 = jacrev(jacrev(log_posterior, argnums=1), argnums=1)
    
    def grad_logp(p, y):
        q = p.clone()
        q[0] = log_grad_p0(p[0], p[1], y[0])
        q[1] = log_grad_p1(p[0], p[1], y[0])
        return q

    def hess_logp(p, y):
        q = p.clone()
        q[0] = log_hess_p0(p[0], p[1], y[0])
        q[1] = log_hess_p1(p[0], p[1], y[0])
        return q

    return grad_logp, hess_logp


def experiment_bone(X, Y):
    grad_logp, hess_logp = get_grad_funcs()

    reg = SWGBoost(grad_logp, hess_logp, DecisionTreeRegressor,
        learner_param = {"criterion": "friedman_mse", 'max_depth': 1, 'random_state': 1},
        learning_rate = 0.1,
        n_estimators = 500,
        n_particles = 10,
        d_particles = 2,
        bandwidth = 0.1)
    reg.fit(X, Y)

    X_test = np.linspace(8, 28, 200).reshape(-1, 1)
    P_test = reg.predict(X_test)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.set_xlim(X.min()-0.5, X.max()+0.5)
    ax.set_ylim(-0.12, 0.28)
    ax.set_xlabel(r"$x$", fontsize=18)
    ax.set_ylabel(r"$y$", fontsize=18)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    sns.scatterplot(x=X.flatten(), y=Y.flatten(), s=20, color="gray", alpha=0.75)
    for i in range(P_test.shape[1]):
        sns.lineplot(x=X_test.flatten(), y=P_test[:,i,0].flatten(), color="red", linewidth=2, alpha=1.0, ax=ax)

    fig.tight_layout()
    fig.savefig("../result/01_cde_plot_bone_m.png", dpi=288)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.set_xlim(X.min()-0.5, X.max()+0.5)
    ax.set_ylim(-0.12, 0.28)
    ax.set_xlabel(r"$x$", fontsize=18)
    ax.set_ylabel(" ", fontsize=18)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    sns.scatterplot(x=X.flatten(), y=Y.flatten(), s=20, color="gray", alpha=0.75)

    X_eval = np.array([12.5, 18, 22])
    Y_plot = np.linspace(-0.1, 0.26, 200)
    for X_eval_ith in X_eval:
        P_plot = reg.predict(X_eval_ith.reshape(-1, 1))
        predictive = np.mean(stats.norm.pdf(Y_plot.reshape(-1, 1), loc=P_plot[0,:,0], scale=np.exp(P_plot[0,:,1])), axis=1)
        ax.vlines(X_eval_ith, ymin=-0.2, ymax=0.5, linewidth=1, linestyle="--", color="gray", alpha=1.0)
        ax.plot(X_eval_ith + predictive.flatten()*0.15, Y_plot, color="red", linewidth=2)

    fig.tight_layout()
    fig.savefig("../result/01_cde_plot_bone_y.png", dpi=288)


def experiment_geyser(X, Y):
    grad_logp, hess_logp = get_grad_funcs()

    reg = SWGBoost(grad_logp, hess_logp, DecisionTreeRegressor,
        learner_param = {"criterion": "friedman_mse", 'max_depth': 1, 'random_state': 1},
        learning_rate = 0.1,
        n_estimators = 500,
        n_particles = 10,
        d_particles = 2,
        bandwidth = 0.1)
    reg.fit(X, Y)

    X_test = np.linspace(40, 100, 200).reshape(-1, 1)
    P_test = reg.predict(X_test)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.set_xlim(X.min()-2, X.max()+2)
    ax.set_ylim(0.6, 6.4)
    ax.set_xlabel(r"$x$", fontsize=18)
    ax.set_ylabel(r"$y$", fontsize=18)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    sns.scatterplot(x=X.flatten(), y=Y.flatten(), s=20, color="gray", alpha=0.75)
    for i in range(P_test.shape[1]):
        sns.lineplot(x=X_test.flatten(), y=P_test[:,i,0].flatten(), color="red", linewidth=2, alpha=1.0, ax=ax)

    fig.tight_layout()
    fig.savefig("../result/01_cde_plot_geyser_m.png", dpi=288)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.set_xlim(X.min()-2, X.max()+2)
    ax.set_ylim(0.6, 6.4)
    ax.set_xlabel(r"$x$", fontsize=18)
    ax.set_ylabel(" ", fontsize=18)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    sns.scatterplot(x=X.flatten(), y=Y.flatten(), s=20, color="gray", alpha=0.75)

    X_eval = np.array([55.5, 70.5, 84.5])
    Y_plot = np.linspace(1, 6, 200)
    for X_eval_ith in X_eval:
        P_plot = reg.predict(X_eval_ith.reshape(-1, 1))
        predictive = np.mean(stats.norm.pdf(Y_plot.reshape(-1, 1), loc=P_plot[0,:,0], scale=np.exp(P_plot[0,:,1])), axis=1)
        ax.vlines(X_eval_ith, ymin=1, ymax=6, linewidth=1, linestyle="--", color="gray", alpha=1.0)
        ax.plot(X_eval_ith + predictive.flatten()*4.0, Y_plot, color="red", linewidth=2)

    fig.tight_layout()
    fig.savefig("../result/01_cde_plot_geyser_y.png", dpi=288)



def main():
    np.random.seed(1)
    torch.manual_seed(1)
    torch.set_num_threads(1)

    df = pd.read_csv("../data/cde/bmd.tsv", sep='\t')
    X = df["age"].to_numpy().reshape(-1, 1)
    Y = df["spnbmd"].to_numpy().reshape(-1, 1)
    experiment_bone(X, Y)
    
    df = pd.read_csv("../data/cde/faithful.csv", index_col=0)
    X = df["waiting"].to_numpy().reshape(-1, 1)
    Y = df["eruptions"].to_numpy().reshape(-1, 1)
    experiment_geyser(X, Y)



#==========================================
# Execution
#==========================================
if __name__ == "__main__":
    main()


