#!/bin/bash
# Run task-01.ipynb top to bottom on an Amarel GPU node and save the executed notebook.
#
# One-time setup on a login node, from this assignment/ folder:
#   uv sync                                   # build the .venv (several GB of PyTorch)
#   uv run python -c "import torchvision; torchvision.datasets.CIFAR10('data/raw', download=True)"
#   mkdir -p runs                             # Slurm needs this before the job starts
#
# Submit (from this assignment/ folder):
#   sbatch train_model.slurm
#
# Monitor:
#   squeue -u $USER
#   tail -f runs/run_notebook_<jobid>.out
#
# Output: task-01.executed.ipynb next to the original, with every cell's
# output (prints, plots) filled in. Your task-01.ipynb is left untouched.
# If a cell raises an error, execution stops and the traceback is in the log.
#
#SBATCH --job-name=run_notebook
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=runs/%x_%j.out

set -euo pipefail

# Batch jobs don't always load your login profile, so make sure uv is on PATH.
export PATH="$HOME/.local/bin:$PATH"

# Have PyTorch use exactly the cores Slurm allocated, not every core on the node.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$SLURM_SUBMIT_DIR"

# The notebook saves weights to models/, which git doesn't track, so a fresh clone won't have it.
mkdir -p models

echo "Job $SLURM_JOB_ID on $(hostname), $SLURM_CPUS_PER_TASK CPUs, started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
uv run python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

# timeout=-1 disables nbconvert's per-cell time limit so the training cell isn't killed.
uv run jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=-1 \
    --output task-01.executed.ipynb \
    task-01.ipynb

echo "Finished $(date)"
