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
import scipy.stats as stats
import scipy.special as special
from sklearn.metrics import average_precision_score



#==========================================
# Main Classes
#==========================================
class Gaussian_Loc_Scale:
    def __init__(self):
        self.tmp = 0
    
    def pdf_each(self, y: np.array, p: np.array) -> np.array:
        l = stats.norm.pdf(y, loc=p[:,0], scale=np.exp(p[:,1]))
        return np.mean(l)
    
    def pdf(self, Y: np.array, P: np.array) -> np.array:
        return np.array( [ self.pdf_each(Y[i], P[i]) for i in range(P.shape[0]) ] ).reshape(-1, 1)
    
    def logpdf(self, Y: np.array, P: np.array) -> np.array:
        return np.log( self.pdf(Y, P) )
     
    def negative_loglikelihood(self, Y: np.array, P: np.array):
        return - np.mean( self.logpdf(Y, P) )
    
    def rmse(self, Y: np.array, P: np.array) -> np.array:
        Y_pred = np.mean(P[:,:,0], axis=1)
        return np.sqrt( np.mean( ( Y - Y_pred )**2 ) )
    
    def negative_loglikelihood_with_standardised_output(self, Z: np.array, P: np.array, Y_std: float):
        # change of variable: Z is standardised ouput and Y is original output
        # i.e. Z = ( Y - Y_train_mean ) / Y_train_std <==> Y = Y_train_mean + Y_train_std * Z
        # the likelihood of Y will be P(Y) = P(Z) * |dZ/dY| = P(Z) * (1/Y_train_std)
        return - np.mean( self.logpdf(Z, P) ) + np.log(Y_std)
    
    def rmse_with_standardised_output(self, Z: np.array, P: np.array, Y_std: float) -> np.array:
        # change of variable: Z is standardised ouput and Y is original output
        # i.e. Z = ( Y - Y_train_mean ) / Y_train_std <==> Y = Y_train_mean + Y_train_std * Z
        # the difference of Y will be ( Y_train_mean + Y_train_std * Z ) - ( Y_train_mean + Y_train_std * Z_pred) = Y_train_std * (Z - Z_pred)
        Z_pred = np.mean( P[:,:,0] , axis=1 , keepdims=True )
        return Y_std * np.sqrt( np.mean( ( Z - Z_pred )**2 ) )
    
    def sample_each(self, p: np.array, num: int) -> np.array:
        randid = np.random.randint(p.shape[0], size=num)
        sample = stats.norm.rvs(loc=p[:,0], scale=np.exp(p[:,1]), size=(num, p.shape[0])).T
        return sample[randid, np.arange(randid.size)]
    
    def sample(self, P: np.array, sample_num: int) -> np.array:
        return np.stack([self.sample_each(p, sample_num) for p in P], axis=0)
    
    def empirical_inverse_cdf(self, P: np.array, value: np.array, sample_num: int = 1000) -> np.array:
        sample = self.sample(P, sample_num)
        return np.quantile(sample, value, axis=1)


class Categorical:
    def __init__(self, num_class: int = 2):
        self.num_class = num_class
    
    def negative_loglikelihood(self, Y: np.array, P: np.array):
        P_new = np.concatenate((P, np.zeros((P.shape[0], P.shape[1], 1))), axis=-1)
        Q_all = np.exp( P_new - special.logsumexp(P_new, axis=-1, keepdims=True) )
        Q = np.mean(Q_all, axis=1)
        return - np.mean( np.sum(np.log(Q) * Y , axis=-1) )
    
    def accuracy(self, Y: np.array, P: np.array):
        P_new = np.concatenate((P, np.zeros((P.shape[0], P.shape[1], 1))), axis=-1)
        Q_all = np.exp( P_new - special.logsumexp(P_new, axis=-1, keepdims=True) )
        Q = np.mean(Q_all, axis=1)
        return 100 * np.mean( np.argmax(Q, axis=-1) == np.argmax(Y, axis=-1) )
    
    def ood_detection(self, P: np.array, P_ood: np.array):
        P_new = np.concatenate((P, np.zeros((P.shape[0], P.shape[1], 1))), axis=-1)
        Q_all = np.exp( P_new - special.logsumexp(P_new, axis=-1, keepdims=True) )
        Q = np.mean(Q_all, axis=1)
        
        P_new_ood = np.concatenate((P_ood, np.zeros((P_ood.shape[0], P_ood.shape[1], 1))), axis=-1)
        Q_all_ood = np.exp( P_new_ood - special.logsumexp(P_new_ood, axis=-1, keepdims=True) )
        Q_ood = np.mean(Q_all_ood, axis=1)

        score_01 = 1 / np.var(Q_all, axis=1).max(axis=1)
        score_00 = 1 / np.var(Q_all_ood, axis=1).max(axis=1)
        labels = np.concatenate([np.ones(Q.shape[0]), np.zeros(Q_ood.shape[0])], axis=0)
        scores = np.concatenate([score_01, score_00], axis=0)

        return 100 * average_precision_score(labels, scores)


