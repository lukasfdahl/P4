#!/bin/bash
#####################
# preprocess_shards.slurm
# ONE-TIME job: pack npy_dir videos into tar shards for fast Ceph streaming.
# No GPU needed — pure CPU/IO job.
#
# Submit:
#   sbatch preprocess_shards.slurm
#
# After this completes, training jobs detect shards automatically.
# Nothing else needs to change in configs or train.py.
#####################

#SBATCH --job-name=pack_shards
#SBATCH --output=pack_shards_%j.out
#SBATCH --error=pack_shards_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

CONTAINER=/ceph/container/pytorch/pytorch_25.09.sif

echo "==========================================================="
echo "Shard packer on: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "==========================================================="

export PATH="$PWD/bin:$HOME/.local/bin:$PATH"

# numpy is the only dep beyond stdlib
singularity exec --nv -B $PWD:$PWD $CONTAINER \
    python3 -m pip install --user numpy

echo ""
echo "Packing dataset/ → dataset/shards/ ..."
singularity exec --nv -B $PWD:$PWD $CONTAINER \
    python3 pack_shards.py \
        --dataset     dataset/ \
        --out         dataset/shards/ \
        --n-per-shard 100 \
        --verify

EXIT_CODE=$?
echo ""
echo "End: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS — shards ready in dataset/shards/"
    echo ""
    echo "Submit training as normal — shards are auto-detected:"
    echo "  sbatch --export=CONFIG=config_P0_baseline.yaml run_train.slurm"
    echo ""
    echo "After confirming training works you can optionally free space:"
    echo "  rm -rf dataset/video_*"
else
    echo "FAILED (exit=$EXIT_CODE) — original npy_dirs are untouched."
fi
exit $EXIT_CODE
