#! /bin/bash

sbatch <<EOT
#! /bin/bash
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH -J $2
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -e $NFS/code/sinf/results/$2/err.txt
#SBATCH -o $NFS/code/sinf/results/$2/out.txt
#SBATCH --tasks-per-node=1

cd $NFS/code/sinf/sinf
MKL_THREADING_LAYER=GNU python utils/eval.py --subject_id=$1 --job_id=$2
exit()
EOT