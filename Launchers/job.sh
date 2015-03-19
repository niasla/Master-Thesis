#!/bin/bash

cd $ZDISK/~ngal/MT/PythonCode
#./train.py -l 2048,2048,2048 -v 1 -e 1   
#./train.py -l 512,512,512 -v 2 -e 100 -fte 100 
#./train.py -l 512,512,512 -v 2 -e 25 -fte 100 
#./train.py -l 2048,2048,2048,2048 -v 2 -e 3 -fte 3 #-tr 100 -dev 50 #-ftonly DBN_RBM_final.save
./train.py -l 512,512,512,512 -v 2 -grbme 225 -rbm 75 -ftonly DBN_RBM_final.save
#./trainsvm.py -v 1 -C 0.5
#./trainsvm.py -linear 1 -v 1 

