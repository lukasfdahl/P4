#!/bin/bash
#####################
# Slurm parameters
#####################

#SBATCH --job-name=modbench_objectdetector
#SBATCH --output=modbench_objectdetector_%j.out
#SBATCH --error=modbench_objectdetector_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=15
#SBATCH --mem=24G
#SBATCH --time=12:00:00

CONTAINER=/ceph/container/pytorch/pytorch_25.09.sif

echo "=========================================================="
echo "Job is running on compute node: $SLURMD_NODENAME"
echo ""
echo "To view MLflow, open a NEW terminal on your LOCAL laptop and run:"
echo "ssh -N -L ${MLFLOW_PORT}:$SLURMD_NODENAME:${MLFLOW_PORT} <USER>@<LOGIN_NODE>"
echo "Then open: http://localhost:${MLFLOW_PORT}"
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
cat > /tmp/constraints.txt << 'EOF'
torch==2.9.0a0+50eac811a6.nv25.09
torchvision==0.24.0a0+98f8b375
numpy<2.3.0
EOF

singularity exec --nv -B $PWD:$PWD $CONTAINER \
    python3 -m pip install --user \
    --constraint /tmp/constraints.txt \
    -r requirements.txt

singularity exec --nv -B $PWD:$PWD $CONTAINER \
    python3 -m pip install --user --no-deps \
    /ceph/project/klnrs/wheels/torch_linear_assignment-0.0.6-cp312-cp312-linux_x86_64.whl

#####################
# 3. Start MLflow in background
#####################
MLFLOW_PORT=$((8500 + SLURM_JOB_ID % 1000))
TBBOARD_PORT=$((9500 + SLURM_JOB_ID % 1000))

echo "Running on node: $SLURMD_NODENAME"
echo "MLflow port:      $MLFLOW_PORT"
echo "TensorBoard port: $TBBOARD_PORT"

singularity exec --nv -B $PWD:$PWD $CONTAINER \
    python3 -m mlflow server \
    --host 0.0.0.0 \
    --port $MLFLOW_PORT \
    --gunicorn-opts "--access-logfile /dev/null" 2>&1 | grep -v "^INFO:" &

sleep 5

#####################
# 4. Run Training
#####################
singularity exec --nv --env MLFLOW_PORT=$MLFLOW_PORT -B $PWD:$PWD $CONTAINER \
    python3 train.py --config config.yaml
