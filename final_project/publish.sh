#!/bin/bash

jupyter nbconvert --to latex 'Final Project.ipynb'
pdflatex 'Final Project.tex'
rm *.log *.out *.aux *.tex
rm -r 'Final Project_files/'
