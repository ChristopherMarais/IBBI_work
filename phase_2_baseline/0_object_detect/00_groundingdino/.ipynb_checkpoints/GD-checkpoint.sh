#!/bin/bash
#SBATCH --job-name=GroundingDINO_Eval    # Job name
#SBATCH --output=/blue/hulcr/eric.kuo/GroundingDINO_Eval/eval_run.out  # Standard output log (%j will be replaced by job ID)
#SBATCH --error=/blue/hulcr/eric.kuo/GroundingDINO_Eval/eval_run.err   # Standard error log (%j will be replaced by job ID)
#SBATCH --mail-type=ALL                  # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=gmarais@ufl.edu     # Your email address
#SBATCH --account=hulcr                  # Account name
#SBATCH --qos=hulcr                      # Quality of Service
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks (usually 1 for a single Python script)
#SBATCH --cpus-per-task=8               # Number of CPU cores per task (adjust as needed)
#SBATCH --mem=64G                        # Memory per node (adjust as needed, e.g., 64G, 128G)
#SBATCH --partition=gpu                  # Partition name (gpu for GPU nodes)
#SBATCH --gpus=a100:1                    # Number and type of GPUs (e.g., a100:1 for one A100 GPU)
#SBATCH --time=72:00:00                  # Time limit hrs:min:sec (adjust based on expected runtime)

# --- User Configuration Section ---
# IMPORTANT: Define the path to your Python script and your Conda environment name here.

# Full path to the directory containing your Python script
SCRIPT_DIR="/blue/hulcr/gmarais/PhD/phase_2_baseline/0_object_detection/00_groundingdino" # <<< CHANGE THIS TO YOUR SCRIPT'S DIRECTORY

# Name of your Conda environment that has the required packages (torch, transformers, etc.)
CONDA_ENV_NAME="EEL" # <<< CHANGE THIS TO YOUR CONDA ENVIRONMENT NAME

# Name of your Python script
PYTHON_SCRIPT_NAME="GroundingDINO.py" # <<< CHANGE THIS IF YOUR SCRIPT HAS A DIFFERENT NAME
# --- End User Configuration Section ---

# Ensure the output directory for SLURM logs exists
mkdir -p /blue/hulcr/eric.kuo/GroundingDINO_Eval

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "CONDA_ENV_NAME: $CONDA_ENV_NAME"
echo "PYTHON_SCRIPT_NAME: $PYTHON_SCRIPT_NAME"

# Purge modules to start with a clean environment
module purge

# Load Conda module (the specific command might vary slightly on different clusters)
module load conda # Or `module load anaconda` or `module load miniconda` etc.

# Activate the Conda environment
echo "Activating Conda environment: $CONDA_ENV_NAME"
source activate "$CONDA_ENV_NAME" # Some systems might use `conda activate "$CONDA_ENV_NAME"`

# Check if conda activation was successful
if [ $? -ne 0 ]; then
  echo "Error: Failed to activate Conda environment '$CONDA_ENV_NAME'."
  exit 1
fi
echo "Conda environment activated."
which python

# Change to the specific directory where your Python script is located
echo "Changing directory to $SCRIPT_DIR"
cd "$SCRIPT_DIR"
if [ $? -ne 0 ]; then
  echo "Error: Failed to change directory to '$SCRIPT_DIR'."
  exit 1
fi
echo "Current directory: $(pwd)"

# Run the Python file
echo "Running Python script: $PYTHON_SCRIPT_NAME"
python "$PYTHON_SCRIPT_NAME"

# Deactivate Conda environment (optional, but good practice)
conda deactivate

echo "Job finished at $(date)"
