#!/bin/bash

jupyter nbconvert --to latex 'Random Forest, XGBoost, and SVM.ipynb'
pdflatex 'Random Forest, XGBoost, and SVM.tex'
rm *.log *.out *.aux *.tex
rm -r 'Random Forest, XGBoost, and SVM_files'/
