#!/bin/bash/

jupyter nbconvert --to latex RF_XGB_SVM.ipynb
pdflatex RF_XGB_SVM.tex
rm RF_XGB_SVM.log RF_XGB_SVM.out RF_XGB_SVM.aux RF_XGB_SVM.tex
