#==========================================
# Header
#==========================================
# Copyright (c) Takuo Matsubara
# All rights reserved.
# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.



#==========================================
# Import Library
#==========================================
import numpy as np
import pandas as pd
import torch
from torch.func import jacrev
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, '../src')
from fwgboost import FWGBoost
from sklearn.kernel_ridge import KernelRidge



#==========================================
# Main Functions
#==========================================
def plot_output(X_test, P_test, filename, is_yon=False, scale=0.5):
    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-2.4, 2.4)
    ax.set_xlabel(r"$x$", fontsize=20)
    ylabel = r"$\theta$" if is_yon else " "
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    cil = np.sin(X_test).flatten() + 1.96 * scale
    ciu = np.sin(X_test).flatten() - 1.96 * scale
    ax.fill_between(X_test.flatten(), cil, ciu, color='b', alpha=0.1)
    
    for ith in range(P_test.shape[1]):
        sns.lineplot(x=X_test.flatten(), y=P_test[:,ith].flatten(), linewidth=2, color="red", ax=ax)

    fig.tight_layout()
    fig.savefig("./result/output_"+filename+".png", dpi=288)


def main():
    np.random.seed(1)
    torch.manual_seed(1)
    torch.set_num_threads(1)
    
    X = np.linspace(-3.5, 3.5, 10).reshape(-1,1)
    Z = np.sin(X)

    scale = 0.5
    grad_logp = lambda p, y: - (p - y) / scale**2
    init_locs = np.linspace(-0.4, 0.4, 10).reshape(-1, 1)

    reg = FWGBoost(grad_logp, KernelRidge,
        learner_param = {'kernel': 'rbf', 'alpha': 0.0, 'gamma': 0.25},
        learning_rate = 0.05,
        n_estimators = 100,
        n_particles = 10,
        d_particles = 1,
        bandwidth = 0.1,
        init_iter = 0,
        init_locs = init_locs)
    reg.fit(X, Z)
    
    X_test = np.linspace(-3.5, 3.5, 500).reshape(-1,1)
    P_test0 = np.repeat(init_locs, X_test.shape[0], axis=1).T
    P_tests = reg.predict_eachitr(X_test).reshape((reg.n_estimators, X_test.shape[0], reg.n_particles))

    plot_output(X_test, P_test0, "M_000", is_yon=True)
    plot_output(X_test, P_tests[14], "M_015")
    plot_output(X_test, P_tests[99], "M_100")



#==========================================
# Execution
#==========================================
if __name__ == "__main__":
    main()


