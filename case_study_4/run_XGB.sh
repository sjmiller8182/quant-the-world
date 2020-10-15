#!/usr/bin/env bash
#SBATCH -J XGB
#SBATCH -o XGB-o.txt
#SBATCH -e XGB-e.txt
#SBATCH -p standard-mem-s --mem=20G
#SBATCH -n 36
#SBATCH -s

python ./xgboost_train.py
