#!/bin/bash
#
#SBATCH --job-name=symbiosis_finetune
#SBATCH --output=finetune_job.out
#SBATCH --error=finetune_job.err
#SBATCH --partition=test
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

# Load the necessary modules (MERCED specific)
module load python/3.11.0

# Activate your Python virtual environment
source /scratch/pperez40/symbiosis_llm/symbio_env/bin/activate

# Run your Python script
python /scratch/pperez40/symbiosis_llm/python_scripts/finetune_model.py
