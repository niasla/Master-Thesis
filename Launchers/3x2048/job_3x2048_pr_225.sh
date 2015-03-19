#!/bin/bash

cd $PCODE
./train.py -l 2048,2048,2048 -v 2 -grbme 225 -rbme 75 -prfreq 25 -ftfreq 25 -maxbatch 140
