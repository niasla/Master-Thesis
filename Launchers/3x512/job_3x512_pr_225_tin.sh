#!/bin/bash

cd $ZDISK/~ngal/MT/PythonCode
./train.py -l 512,512,512 -v 2 -grbme 225 -rbme 75 -prfreq 25 -ftfreq 25 -maxbatch 180 -tin 1
