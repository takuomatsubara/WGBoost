#==========================================
# Header
#==========================================
# Copyright (c) Takuo Matsubara
# All rights reserved.
# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.



#==========================================
# Import Library
#==========================================
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge



#==========================================
# Main Functions
#==========================================
learner_loader = {
    "tree": {
        "class": DecisionTreeRegressor,
        "default_param": { 'criterion': 'friedman_mse', 'max_depth': 3, 'random_state': 1 }
    },
    "linear": {
        "class": LinearRegression,
        "default_param": { 'random_state': 1 }
    },
    "kernel": {
        "class": KernelRidge,
        "default_param": { 'random_state': 1 }
    }
}


