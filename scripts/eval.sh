#! /bin/bash

sbatch <<EOT
#! /bin/bash
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH -J $1
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -e $NFS/code/sinf/results/$1/err.txt
#SBATCH -o $NFS/code/sinf/results/$1/out.txt
#SBATCH --tasks-per-node=1

cd $NFS/code/sinf/sinf
MKL_THREADING_LAYER=GNU python evaluate/atlas_as_bridge.py -j=$1
exit()
EOT