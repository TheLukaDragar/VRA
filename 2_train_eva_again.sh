#!/bin/sh
#SBATCH --job-name=train_convnext
#SBATCH --output=train_convnext_%j.out
#SBATCH --error=train_convnext_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --mem=0
#SBATCH --cpus-per-task=12
#SBATCH --time=0-10:00:00
#SBATCH --partition=gpu

#example salloc --nodes=1 --gres=gpu:2 --ntasks-per-node=2 --mem=0 --time=0-10:00:00 --cpus-per-task=12 --job-name=Interactive_GPU2 --partition=gpu 



source  /d/hpc/projects/FRI/ldragar/miniconda3/etc/profile.d/conda.sh
conda activate /d/hpc/projects/FRI/ldragar/pytorch_env

seed=-1 #no seed better results
cp_id="wrc2h54x" #our best eva model
export WANDB__SERVICE_WAIT=300
#script is made to run on 1 node with 2 gpus
srun --nodes=1 --exclusive --gpus=2 --ntasks-per-node=2 --time=0-10:00:00 -p gpu python train_eva_again.py --seed $seed --cp_id $cp_id