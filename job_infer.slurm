#!/bin/bash
#SBATCH --job-name=bendr-inference-smoke-test
#SBATCH --output=logs/bendr_inference_%j.out
#SBATCH --error=logs/bendr_inference_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00

# Load Anaconda module
module load anaconda3/2023.03

# Directly run python from your conda environment's bin path to avoid shell hook issues
PYTHON_BIN="/home/hice1/rkrishnan72/.conda/envs/bendr-env/bin/python"

echo "Running BENDR inference smoke test on PACE GPU node..."
$PYTHON_BIN infer_smoke_test.py
echo "Job complete."
