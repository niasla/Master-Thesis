#!/bin/bash
#C=0.5
#for C in 0.5 1.0
#  do
for layer in 3 4
  do
  for layer_sz in 512 1024 2048
    do
    for bl in $(seq 1 $layer)
      do
      qsub -v blayer=$bl,lay=$layer,lay_sz=$layer_sz -l nodes=1,walltime=24:00:00 ./testlinearsvmbl.sh
    done
  done
done
#done

#all mfcc
qsub -l nodes=1,walltime=24:00:00 ./testlinearsvmmfcc.sh

#qsub -l nodes=1,walltime=24:00:00 ./testsvm.py -v 1 -C 1.0 -linear 1