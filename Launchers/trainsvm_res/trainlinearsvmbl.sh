#!/bin/bash

layers=$lay_sz,$lay_sz,$lay_sz

if [ "$lay" = "4" ]; then
    layers=$lay_sz,$lay_sz,$lay_sz,$lay_sz
fi

if [ "$lay" = "5" ]; then
    layers=$lay_sz,$lay_sz,$lay_sz,$lay_sz,$lay_sz
fi

if [ "$lay" = "2" ]; then
    layers=$lay_sz,$lay_sz
fi



cd $ZDISK/~ngal/MT/PythonCode
./trainsvm.py -linear 1 -v 1 -l $layers -bl $blayer -e 225
