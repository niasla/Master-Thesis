#!/bin/bash

for bl in $(seq 1 4)
  do
  qsub -v blayer=$bl -l nodes=1,walltime=24:00:00 ./testlinearsvmbl_tin.sh
done

#all mfcc
#qsub -l nodes=1,walltime=24:00:00 ./testlinearsvmmfcc.sh

#qsub -l nodes=1,walltime=24:00:00 ./testsvm.py -v 1 -C 1.0 -linear 1 