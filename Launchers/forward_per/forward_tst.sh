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

#./test.py -v 1 -e 225 -l $layers -save 1 -training 1
./test.py -v 1 -e 225 -l $layers -per 1
