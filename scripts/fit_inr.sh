#! /bin/bash
if [ ! -d $NFS/code/sinf/results/$1 ]
then
    mkdir $NFS/code/sinf/results/$1
fi

cd $NFS/code/sinf/sinf
MKL_THREADING_LAYER=GNU python3 fit_inr.py -j=$1 -c=$2 -s=${3:-None}
