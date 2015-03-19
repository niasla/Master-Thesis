for layer in 2
  do
  for layer_sz in 1024 2048
    do
    qsub -v lay=$layer,lay_sz=$layer_sz -l nodes=1,walltime=24:00:00 ./train_hmm.sh
  done
done