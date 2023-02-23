#!/bin/bash
#SBATCH --job-name=p_tune_tp2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2

#SBATCH --cpus-per-task=32
#SBATCH --mem=1024gb

#SBATCH --time=50:00:00
#SBATCH --output=p_tune_tp2_%j.out
#SBATCH --partition=hpg-ai

#SBATCH --exclude=c1000a-s11

module load singularity

pwd; hostname; date

export NCCL_NET_PLUGIN=none
export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1 

mkdir -p $PWD/tmp2

export SINGULARITYENV_TMPDIR=$PWD/tmp2 
export SLURM_TMPDIR=$PWD/tmp2
export TMPDIR=$PWD/tmp2

export WORLD_SIZE=2 ##
CONTAINER=/blue/ufhpc/hityangsir/test-nemo/debug2/nemo:22.11

DEBUG_SLURM=false
if "$DEBUG_SLURM"; then
    set | grep SLURM | while read line; do echo "# $line"; done
fi

#export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

echo "################################################################"
nvidia-smi
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
echo "################################################################"
echo


srun singularity exec --nv --bind $PWD:/work $CONTAINER \
    python3 /work/debug_ptune.py \
    --config-name=p_tuning_config_tp2.yaml
