#!/usr/bin/env bash
#SBATCH -J RFsearch
#SBATCH -o RF_search-o.txt
#SBATCH -e RF_search-e.txt
#SBATCH -p standard-mem-s --mem=30G
#SBATCH -n 36
#SBATCH -s

python ./RF_search.py
