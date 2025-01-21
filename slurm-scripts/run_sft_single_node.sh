#!/bin/bash
HOST=$1
NODES=$2

cd ${WORK}/OLMo

export NCCL_DEBUG=INFO
export NODENAME=$(hostname -s)

export RANK=$SLURM_PROCID
export FS_LOCAL_RANK=$SLURM_PROCID
export LOCAL_WORLD_SIZE=1 # $SLURM_NTASKS_PER_NODE
export LOCAL_RANK=0 # $SLURM_LOCALID
export NODE_RANK=$((($RANK - $LOCAL_RANK) / $LOCAL_WORLD_SIZE))

echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "NODE_RANK:= "  $NODE_RANK
echo "WORLD_SIZE:= "  $WORLD_SIZE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

# ******************* These are read internally it seems ***********************************
# ******** Master port, address and world size MUST be passed as variables for DDP to work 
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$SLURM_NNODES
echo "MASTER_PORT"=$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
# ******************************************************************************************

# zoom zoom - recommended from lightning
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32

# for debugging
export NCCL_DEBUG=INFO

save_root=${SCRATCH}/checkpoints/sft
path_to_checkpoint="${SCRATCH}/base_models/deepseek/olmo/deepseek-coder-1.3b-base"
t_warmup=20

torchrun --nproc_per_node=1 \
    scripts/train.py \
    configs/astro-sft/DS-Coder-1B.yaml \
    --reset_trainer_state \
    --remote_save_folder=null \
    --save_overwrite \
    --reset_optimizer_state \
    --load_path="${path_to_checkpoint}" \
    --save_folder="${save_root}/SFT-DS-Coder-1B-final" \
    --restart_from_unsharded_checkpoint=True \
    --scheduler.t_warmup=${t_warmup}