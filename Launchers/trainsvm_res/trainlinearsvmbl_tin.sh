#!/bin/bash

cd $ZDISK/~ngal/MT/PythonCode
./trainsvm.py -linear 1 -v 1 -l 512,512,512,512 -bl $blayer -e 225 -tin 1
