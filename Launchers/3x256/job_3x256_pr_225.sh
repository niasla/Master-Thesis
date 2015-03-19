#!/bin/bash

cd $ZDISK/~ngal/MT/PythonCode
./train.py -l 256,256,256 -v 2 -grbme 225 -rbme 75 -prfreq 25 -ftfreq 25 -maxbatch 140
