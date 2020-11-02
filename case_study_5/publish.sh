#!/bin/bash

jupyter nbconvert --to latex 'Russian Housing Data Imputation.ipynb'
pdflatex 'Russian Housing Data Imputation.tex'
rm *.log *.out *.aux *.tex
rm -r 'Russian Housing Data Imputation_files'
