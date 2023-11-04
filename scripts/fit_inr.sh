#! /bin/bash
if [ ! -d $NFS/code/sinf/results/$1 ]
then
    mkdir $NFS/code/sinf/results/$1
fi
sbatch <<EOT
#! /bin/bash
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH -J $1
#SBATCH --gres=gpu:1
#SBATCH -e $NFS/code/sinf/results/$1/err.txt
#SBATCH -o $NFS/code/sinf/results/$1/out.txt
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1

cd $NFS/code/sinf/sinf
MKL_THREADING_LAYER=GNU python fit_inr.py -j=$1 -c=$2 -s=${3:-None} -t=${4:-0}
exit()
EOT
