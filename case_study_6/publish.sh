#!/bin/bash

jupyter nbconvert --to latex 'Searching for the Higgs Boson with Deep Learning.ipynb'
pdflatex 'Searching for the Higgs Boson with Deep Learning.tex'
rm *.log *.out *.aux *.tex
rm -r 'Searching for the Higgs Boson with Deep Learning_files'
