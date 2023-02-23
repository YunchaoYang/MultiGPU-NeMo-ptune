#!/bin/bash
#SBATCH --job-name=p_tune_training_tp1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=32
#SBATCH --mem=1024gb

#SBATCH --time=50:00:00
#SBATCH --output=p_tune_tp1_%j.out
#SBATCH --partition=hpg-ai

#SBATCH --exclude=c1000a-s11 # avoid some node if there is conflict in PORT number

pwd; hostname; date

module load singularity

# Debug flags
export NCCL_NET_PLUGIN=none
export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1 

# monitor the TMP_DIR
mkdir -p $PWD/tmp
export SINGULARITYENV_TMPDIR=$PWD/tmp
export SLURM_TMPDIR=$PWD/tmp
export TMPDIR=$PWD/tmp

export WORLD_SIZE=1 #
CONTAINER=/blue/ufhpc/hityangsir/test-nemo/debug2/nemo:22.11

# Not necessary when using PL
# set | grep SLURM | while read line; do echo "# $line"; done
# export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

echo "################################################################"
nvidia-smi
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
echo "################################################################"
echo

srun singularity exec --nv $CONTAINER \
    python3 /blue/ufhpc/hityangsir/test-nemo/debug2/tp1/MultiGPU-NeMo-ptune-debug/debug_ptune.py


