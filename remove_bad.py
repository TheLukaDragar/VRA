import wandb
import os
# Set the project and sweep information
project_name = "luka_vra"
sweep_id = "fa6r9hym"

# Authenticate to wandb
wandb.login()

# Connect to the specified project
api = wandb.Api()
sweep = api.sweep(path=project_name + "/" + sweep_id)
print("Sweep Name: {}".format(sweep.name))




# Iterate through the sweep runs
sweep_runs = sweep.runs
print("Number of runs: {}".format(len(sweep_runs)))
print("Run IDs:")
for run in sweep_runs:
    print(run.id)

filtered_run_ids = []

for run in sweep_runs:
    # You may need to replace 'finalfinalscore' with the exact name of the metric you're looking for
    if run.summary.get('final_final_score') is not None and run.summary['final_final_score'] < 0.83:
        filtered_run_ids.append(run.id)

# Print the filtered run IDs
print("Run IDs with finalfinalscore lower than 0.83:")
for run_id in filtered_run_ids:
    print(run_id)
    #delete checkpoint rm -rf
    os.system("rm -rf ./convnext_models/" + run_id)

   

