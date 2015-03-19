#!/bin/bash

cd $ZDISK/~ngal/MT/PythonCode
#rm ./Models/3x512_pr_225/DBN_RBM_pr_225_ft_*.save
./test.py -v 1 -e 225 -l 512,512,512 -save 1 -training 1
./test.py -v 1 -e 225 -l 512,512,512 -save 1