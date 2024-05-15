#==========================================
# Header
#==========================================
# Copyright (c) Takuo Matsubara
# All rights reserved.
# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

# Some of this code modified part of the regression_exp.py file in https://github.com/stanfordmlgroup/ngboost that prepares data and performs early stopping of boosting.
# License: Apache License 2.0
# The copy of the original license is attached at the bottom.



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
from model import Gaussian_Loc_Scale
from datasets import dataset_loader
from learners import learner_loader



#==========================================
# Main Functions
#==========================================
def parse_arguments():
    argparser = ArgumentParser()
    argparser.add_argument("--id", type=str, default="02_uci")
    argparser.add_argument("--n_jobs", type=int, default=10)
    argparser.add_argument("--dataset", type=str, default="housing")
    argparser.add_argument("--k_repeat", type=int, default=20)
    argparser.add_argument("--learner", type=str, default="tree")
    argparser.add_argument("--learning_rate", type=float, default=0.1)
    argparser.add_argument("--n_estimators", type=int, default=4000)
    argparser.add_argument("--n_particles", type=int, default=10)
    argparser.add_argument("--d_particles", type=int, default=2)
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


def create_random_repeat(num_data, num_repeat):
    # Follow thw way in split_data_train_test.py file in https://github.com/yaringal/DropoutUncertaintyExps
    folds = []
    for i in range(num_repeat):
        permutation = np.random.choice(range(num_data), num_data, replace=False)
        train_index = permutation[0:round(num_data*0.9)]
        test_index = permutation[round(num_data*0.9):num_data]
        folds.append((train_index, test_index))
    return folds


def get_grad_funcs():
    alpha = 0.01
    beta = 0.01
    scale = 10
    
    def log_posterior(p0, p1, y):
        return - (1/2) * ( (y - p0)**2 ) / torch.exp(2*p1) - (1/2) * (p0**2) / (scale**2) - ( alpha + 1 ) * p1 - beta / torch.exp(p1)

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


def one_fold(X, Y, kth, fold_kth):
    train_index = fold_kth[0]
    test_index = fold_kth[1]
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    # preprocess data by standardisation, where likelihood will be adjusted using change of variable formula
    (X_train, X_test), (X_train_mean, X_train_std) = preprocess_standardisation(X_train, X_test)
    (Y_train, Y_test), (Y_train_mean, Y_train_std) = preprocess_standardisation(Y_train, Y_test)

    X_buf, X_val, Y_buf, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=1)

    model = Gaussian_Loc_Scale()
    grad_logp, hess_logp = get_grad_funcs()

    reg = SWGBoost(grad_logp, hess_logp, learner_loader[args.learner]['class'],
        learner_param = learner_loader[args.learner]['default_param'],
        learning_rate = args.learning_rate,
        n_estimators = args.n_estimators,
        n_particles = args.n_particles,
        d_particles = args.d_particles,
        bandwidth = args.bandwidth,
        subsample = args.subsample)
    reg.fit(X_buf, Y_buf)

    # early stopping for estimation number using the validation set
    P_val_eachitr = reg.predict_eachitr(X_val)
    RMSE_val_eachitr = [ model.rmse_with_standardised_output(Y_val, P_val, Y_train_std) for P_val in P_val_eachitr ]
    bestitr = int( np.argmin(RMSE_val_eachitr) + 1 )
    
    # train the SWGBoost using all the data
    reg = SWGBoost(grad_logp, hess_logp, learner_loader[args.learner]['class'],
        learner_param = learner_loader[args.learner]['default_param'],
        learning_rate = args.learning_rate,
        n_estimators = bestitr,
        n_particles = args.n_particles,
        d_particles = args.d_particles,
        bandwidth = args.bandwidth,
        subsample = args.subsample)
    reg.fit(X_train, Y_train)
    
    # test prediction
    P_test = reg.predict(X_test)
    RMSE_test = model.rmse_with_standardised_output(Y_test, P_test, Y_train_std)
    print("[{:02d}th trial RMSE] Test {:.4f} | Val {:.4f} | Bestitr {:d}".format(kth, RMSE_test, np.min(RMSE_val_eachitr), bestitr))
    return RMSE_test, bestitr


def main(args):
    np.random.seed(1)
    torch.manual_seed(1)
    torch.set_num_threads(1)
    
    # load dataset
    data = dataset_loader[args.dataset]()
    X, Y = data.iloc[:, :-1].values, data.iloc[:, -1].values.reshape(-1,1)
    folds = create_random_repeat(X.shape[0], args.k_repeat)
    print("=== Dataset {:s} | Num {:d} | Model {:s} ===".format(args.dataset, X.shape[0], "normal"))
    
    results = Parallel(n_jobs=args.n_jobs, backend="multiprocessing")(delayed(one_fold)(X, Y, kth, folds[kth]) for kth in range(len(folds)))
    RMSE_tests = np.array(results)

    print("=== RMSE Test Summary {:.4f} +/- {:.4f} ===".format(np.mean(RMSE_tests[:,0]), np.std(RMSE_tests[:,0])))
    pd.DataFrame(RMSE_tests).to_csv("../result/" + args.id + "_test_rmse_" + args.dataset + ".csv", index=False, header=["RMSE", "bestitr"])



#==========================================
# Execution
#==========================================
if __name__ == "__main__":
    args = parse_arguments()
    main(args)



#==========================================
# Apache License 2.0
#==========================================
'''
Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''


