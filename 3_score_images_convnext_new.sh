#!/bin/sh
#SBATCH --job-name=predict_convnext
#SBATCH --output=predict_convnext_%j.out
#SBATCH --error=predict_convnext_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --cpus-per-task=12
#SBATCH --time=0-10:00:00
#SBATCH --partition=gpu

#example salloc --nodes=1 --gres=gpu:2 --ntasks-per-node=2 --mem=0 --time=0-10:00:00 --cpus-per-task=12 --job-name=Interactive_GPU2 --partition=gpu 


       
source /ceph/hpc/data/st2207-pgp-users/ldragar/miniconda3/etc/profile.d/conda.sh
conda activate  /ceph/hpc/data/st2207-pgp-users/ldragar/pytorch_env


out_predictions_dir='./predictions/'
cp_id=$1 


#script is made to run on 1 node with 1 gpu
srun --nodes=1 --exclusive --gpus=1 --ntasks-per-node=1 --time=0-3:00:00 -p gpu python score_convnext_images_new.py --out_predictions_dir $out_predictions_dir --cp_id $cp_id
