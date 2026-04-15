#!/bin/bash

#SBATCH --job-name=evaluate_labels    # A descriptive name for your job
#SBATCH --output=slurm-%j.out   # File to capture both standard output and error
#SBATCH --error=slurm-%j.out    # Redirect standard error to the same file as output
#SBATCH --time=01:15:00         
#SBATCH --gpus=h200:1
#SBATCH --mem=50G               # Request 50GB of memory
#SBATCH --cpus-per-task=8       # Request 8 CPU cores
#SBATCH --array=4-24%1           # Request 25 jobs in array with a maximum of 1 running at a time

# Change to the directory from which the job was submitted
cd /home/ammonbro/LabelEval

uv run -m label_evaluation --config config_template.yaml --layer $SLURM_ARRAY_TASK_ID