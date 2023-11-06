#! /bin/bash

cd $NFS/code/sinf/sinf
MKL_THREADING_LAYER=GNU python3 evaluate/atlas_as_bridge.py -j=$1 -n=$2