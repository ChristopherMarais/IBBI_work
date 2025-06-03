#!/bin/bash
#SBATCH --job-name=CODETR_5Fold
#SBATCH --output=./slurm_logs/CODETR_5fold_main_%j.out
#SBATCH --error=./slurm_logs/CODETR_5fold_main_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gmarais@ufl.edu # Update with your email if different
#SBATCH --account=npadillacoreano            # Ensure this is your correct account
#SBATCH --qos=npadillacoreano                # Ensure this is your correct QOS
#SBATCH --nodes=1                  # Running all 5 processes on a single node
#SBATCH --ntasks=1                 # The python script itself is one main task that spawns others
#SBATCH --cpus-per-task=12         # Increased CPUs for 5 parallel folds (5*(3 workers + 1 main) = 20, + buffer)
                                   # If using fewer GPUs (e.g., 3), you can reduce this (e.g., to 16)
#SBATCH --mem=128G                 # Total memory for the job (shared by all 5 python processes)
#SBATCH --partition=gpu
#SBATCH --gpus=a100:5              # Requesting 5 A100 GPUs for 5 folds.
                                   # If you only have 3, change to a100:3. Python script will adapt.
#SBATCH --time=72:00:00            # Max walltime

# Ensure the output directory for Slurm logs exists
SLURM_LOG_DIR="./slurm_logs"
mkdir -p ${SLURM_LOG_DIR}

echo "Starting codetr 5-Fold Training Script"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Assigned GPUs: $CUDA_VISIBLE_DEVICES"
echo "Current working directory: $(pwd)"

module purge
module load conda # Or your specific module for loading conda/python environments

# Activate the conda environment
# IMPORTANT: Replace 'YOLO10' with the actual name of your conda environment if different
CONDA_ENV_NAME="EEL" # Based on your previous logs, it seems to be "EEL"
echo "Activating conda environment: ${CONDA_ENV_NAME}"
conda activate ${CONDA_ENV_NAME}
if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment: ${CONDA_ENV_NAME}"
    exit 1
fi
echo "Conda environment activated."
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available to PyTorch: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of GPUs PyTorch sees: $(python -c 'import torch; print(torch.cuda.device_count())')"


# Change to the directory where your python script and results folder should be
# IMPORTANT: Update this path if your codetr_train_5_folds.py is elsewhere
SCRIPT_DIR="/blue/hulcr/gmarais/PhD/IBBI_work/phase_2_baseline/0_object_detect/2_codetr/train/Co-DETR-main" # Assuming script is here
cd ${SCRIPT_DIR}
echo "Changed directory to: $(pwd)"

# Run the Python file
# Ensure the script is named codetr_train_5_folds.py or update the name here
echo "Running Python script: codetr_train_5_folds.py"
python codetr_5fold.py

echo "Python script finished."
echo "End of Slurm job."