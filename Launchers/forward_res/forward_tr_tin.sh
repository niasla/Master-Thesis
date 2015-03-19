#!/bin/bash

#layers=$lay_sz,$lay_sz,$lay_sz

#if [ "$lay" = "4" ]; then
#    layers=$lay_sz,$lay_sz,$lay_sz,$lay_sz
#fi

#if [ "$lay" = "5" ]; then
#    layers=$lay_sz,$lay_sz,$lay_sz,$lay_sz,$lay_sz
#fi

#if [ "$lay" = "2" ]; then
#    layers=$lay_sz,$lay_sz
#fi





cd $ZDISK/~ngal/MT/PythonCode

./test.py -v 1 -e 225 -l 512,512,512,512 -save 1 -training 1 -tin 1

