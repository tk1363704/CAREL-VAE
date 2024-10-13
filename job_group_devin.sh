#!/bin/bash

#SBATCH -t 7-00:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --partition=A100


# Memory usage (MB)
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=300000

#SBATCH --mail-user=devin.hua@monash.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# IMPORTANT!!! check the job name!
#SBATCH -o log/%J-main_all.out
#SBATCH -e log/%J-main_all.err
#
#
#
#
#SBATCH -J CCU_emotion

module load anaconda
export CONDA_ENVS=/nfsdata/data/devinh/envs
source activate $CONDA_ENVS/ecpe
module load cuda/cuda-11.1.0
module load cudnn/cudnn-8.0.4
cd /nfsdata/data/devinh/ECPE_code
CUDA_VISIBLE_DEVICES=0 python drl_classifier_ec_mmd_final_mul_newsplit_emnlp.py



