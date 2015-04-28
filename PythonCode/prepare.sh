#!/bin/bash
#PYTHONPATH=$PYTHONPATH:$(pwd)/DeepLearningTB:/lib64/python2.7/site-packages:/lib/python2.7/site-packages
#export PYTHONPATH

#  0 - async   , 1  - synced

if [ "$1" = "release" ]
then
    CUDA_LAUNCH_BLOCKING=0
    export CUDA_LAUNCH_BLOCKING
    cat ~/.theanorc | awk -F"=" '{if ($1 == "profile"){ print "profile=False"} else {print $0}}'> ~/.theanorc2
elif  [ "$1" = "profile" ]
then
    CUDA_LAUNCH_BLOCKING=1
    export CUDA_LAUNCH_BLOCKING
    cat ~/.theanorc | awk -F"=" '{if ($1 == "profile"){ print "profile=True"} else {print $0}}'> ~/.theanorc2
fi

mv ~/.theanorc2 ~/.theanorc
