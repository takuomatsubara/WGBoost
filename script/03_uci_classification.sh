#!/bin/bash

n_jobs=5
file=../experiment/03_uci_classification.py

python $file --dataset segment --n_jobs $n_jobs
python $file --dataset sensorless --n_jobs $n_jobs
