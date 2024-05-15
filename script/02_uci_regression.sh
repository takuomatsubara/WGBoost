#!/bin/bash

n_jobs=10
file=../experiment/02_uci_regression_nll.py

python $file --dataset housing --n_jobs $n_jobs
python $file --dataset concrete --n_jobs $n_jobs
python $file --dataset energy --n_jobs $n_jobs
python $file --dataset wine --n_jobs $n_jobs
python $file --dataset yacht --n_jobs $n_jobs
python $file --dataset kin8nm --n_jobs $n_jobs
python $file --dataset naval --n_jobs $n_jobs
python $file --dataset power --n_jobs $n_jobs
python $file --dataset protein --n_jobs 5 --k_repeat 5

file=../experiment/02_uci_regression_rmse.py

python $file --dataset housing --n_jobs $n_jobs
python $file --dataset concrete --n_jobs $n_jobs
python $file --dataset energy --n_jobs $n_jobs
python $file --dataset wine --n_jobs $n_jobs
python $file --dataset yacht --n_jobs $n_jobs
python $file --dataset kin8nm --n_jobs $n_jobs
python $file --dataset naval --n_jobs $n_jobs
python $file --dataset power --n_jobs $n_jobs
python $file --dataset protein --n_jobs 5 --k_repeat 5
