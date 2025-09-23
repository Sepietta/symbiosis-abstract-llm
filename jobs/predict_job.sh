#!/bin/bash
#
#SBATCH --job-name=symbiosis_predict
#SBATCH --output=predict_job.out
#SBATCH --error=predict_job.err
#SBATCH --time=00:30:00
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

# Load the necessary modules
module load python/3.11.0

# Activate your Python virtual environment
source /scratch/pperez40/symbiosis_llm/symbio_env/bin/activate

# Run your Python script
python /scratch/pperez40/symbiosis_llm/python_scripts/predict_gene.py
