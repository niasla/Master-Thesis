#!/bin/bash

layers=$lay_sz,$lay_sz,$lay_sz

if [ "$lay" = "2" ]; then
    layers=$lay_sz,$lay_sz
fi

if [ "$lay" = "4" ]; then
    layers=$lay_sz,$lay_sz,$lay_sz,$lay_sz
fi

if [ "$lay" = "5" ]; then
    layers=$lay_sz,$lay_sz,$lay_sz,$lay_sz,$lay_sz
fi

cd $ZDISK/~ngal/MT/PythonCode
./train.py -v 2 -grbme 225 -rbme 75 -l $layers -fte 0 -ftonly DBN_RBM_final.save 

