#!/bin/bash
#####################
# Slurm parameters
#####################

#SBATCH --job-name=objectdetect
#SBATCH --output=V2_overfit%j.out
#SBATCH --error=V2_overfit%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=15
#SBATCH --mem=24G
#SBATCH --time=12:00:00

CONTAINER=/ceph/container/pytorch/pytorch_25.09.sif 

echo "=========================================================="
echo "Job is running on compute node: $SLURMD_NODENAME"
echo ""
echo "To view MLflow, open a NEW terminal on your LOCAL laptop and run:"
echo "ssh -N -L 8501:$SLURMD_NODENAME:8501 <YOUR_USERNAME>@<YOUR_LOGIN_NODE_IP>"
echo ""
echo "Then open your browser to: http://localhost:8501"
echo "=========================================================="

#####################
# 1. Environment Setup
#####################
PROJECT_BIN="$PWD/bin"
mkdir -p "$PROJECT_BIN"
mkdir -p downloaded_videos
export PATH="$PROJECT_BIN:$HOME/.local/bin:$PATH"

#####################
# 2. Install Python Dependencies
#####################
singularity exec --nv -B $PWD:$PWD $CONTAINER \
    python3 -m pip install --user --no-cache-dir -r requirements.txt

#####################
# 3. Start MLflow in background
#####################
echo "Running on node: $SLURMD_NODENAME"

singularity exec --nv -B $PWD:$PWD $CONTAINER \
    python3 -m mlflow server --host 0.0.0.0 --port 8501 \
    --gunicorn-opts "--access-logfile /dev/null" 2>&1 | grep -v "^INFO:" &

sleep 5

#####################
# 4. Run Training
#####################
singularity exec --nv -B $PWD:$PWD $CONTAINER \
    python3 train.py