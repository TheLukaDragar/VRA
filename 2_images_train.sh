#!/bin/sh
#SBATCH --job-name=train_VRA
#SBATCH --output=train_VRA%j.out
#SBATCH --error=train_VRA%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --cpus-per-task=12
#SBATCH --time=1-10:00:00
#SBATCH --partition=gpu

#example salloc --nodes=1 --gres=gpu:2 --ntasks-per-node=2 --mem=0 --time=0-10:00:00 --cpus-per-task=12 --job-name=Interactive_GPU2 --partition=gpu 


       
source /ceph/hpc/data/st2207-pgp-users/ldragar/miniconda3/etc/profile.d/conda.sh
conda activate  /ceph/hpc/data/st2207-pgp-users/ldragar/pytorch_env

wandb_agent='ldragar/convnext/47qukgop'
#script is made to run on 1 node with 1 gpu
srun --nodes=1 --exclusive --gpus=4 --ntasks-per-node=4 --cpus-per-task=12 --time=1-10:00:00 -p gpu wandb agent $wandb_agent
