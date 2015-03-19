#!/bin/bash

for bl in $(seq 1 4)
  do
  qsub -v blayer=$bl -l nodes=1,walltime=24:00:00 ./trainlinearsvmbl_tin.sh
done

