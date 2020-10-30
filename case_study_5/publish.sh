#!/bin/bash

jupyter nbconvert --to latex 'case-study-10.ipynb'
pdflatex 'case-study-10.tex'
# rm *.log *.out *.aux *.tex
