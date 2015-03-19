for layer in 2 3 4 5 
  do
  for layer_sz in 512 1024 2048
    do
    #qsub -v lay=$layer,lay_sz=$layer_sz -l nodes=1,walltime=00:20:00 ./forward_tst.sh
    #qsub -v lay=$layer,lay_sz=$layer_sz -l nodes=1,walltime=00:20:00 ./forward_tr.sh
    qsub -v lay=$layer,lay_sz=$layer_sz -l nodes=1,walltime=00:20:00 ./forward_tst_rl.sh
    qsub -v lay=$layer,lay_sz=$layer_sz -l nodes=1,walltime=00:20:00 ./forward_tr_rl.sh
  done
done