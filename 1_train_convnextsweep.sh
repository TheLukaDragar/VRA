# #!/bin/bash
#SBATCH --array=1-2   #e.g. 1-4 will create agents labeled 1,2,3,4
#NUMBER OF AGENTS TO REGISTER AS WANDB AGENTS
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=2 # must equal number of gpus, as required by Lightning
#SBATCH --mem=0
#SBATCH --time=0-10:00:00
#SBATCH --partition=gpu
#SBATCH --job-name=train_convnext
#SBATCH --output=train_convnext_%j.out
#SBATCH --error=train_convnext_%j.err





source  /d/hpc/projects/FRI/ldragar/miniconda3/etc/profile.d/conda.sh
conda activate /d/hpc/projects/FRI/ldragar/pytorch_env



# SET SWEEP_ID HERE. Note sweep must already be created on wandb before submitting job
SWEEP_ID="eu19pilb"
API_KEY="242d18971b1b7df61cceaa43724c3b1e6c17c49c"

# LOGIN IN ALL TASKS
srun wandb login $api_key

# adapted from https://stackoverflow.com/questions/11027679/capture-stdout-and-stderr-into-different-variables
# RUN WANDB AGENT IN ONE TASK
{
    IFS=$'\n' read -r -d '' SWEEP_DETAILS; RUN_ID=$(echo $SWEEP_DETAILS | sed -e "s/.*\[\([^]]*\)\].*/\1/g" -e "s/[\'\']//g")
    IFS=$'\n' read -r -d '' SWEEP_COMMAND;
} < <((printf '\0%s\0' "$(srun --ntasks=1 wandb agent --count 1 $SWEEP_ID)" 1>&2) 2>&1)

echo "RUN_ID: $RUN_ID"
echo "SWEEP_COMMAND: $SWEEP_COMMAND"

SWEEP_COMMAND="${SWEEP_COMMAND} --wandb_resume_version ${RUN_ID}"

# WAIT FOR ALL TASKS TO CATCH UP
wait

# RUN SWEEP COMMAND IN ALL TASKS
srun  $SWEEP_COMMAND