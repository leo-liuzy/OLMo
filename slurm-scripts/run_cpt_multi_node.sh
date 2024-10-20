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

save_root=${SCRATCH}/checkpoints/tenk_dump_package_dedup
data_folder=${SCRATCH}/processed_data/sample_dump_package/dedup
data_folder=${SCRATCH}/processed_data/tenk_dump_package/dedup
path_to_checkpoint="${SCRATCH}/base_models/OLMo/OLMo-1B-final"

# no_biblio_mds,11 no_html_mds,8 no_tab_fig_mds,9 cat_mds,14 # sample bs=8

# no_biblio_mds,132 no_html_mds,67 no_tab_fig_mds,102 cat_mds,155 # 10k bs=8

# no_biblio_mds,4 no_html_mds,2 no_tab_fig_mds,3 cat_mds,4 # 10k bs=2048
filter_warmup_pair=cat_mds,4

echo $filter_strategy
echo $t_warmup

IFS=, read -r filter_strategy t_warmup <<< $filter_warmup_pair
torchrun --nproc_per_node=1 \
    --rdzv-backend=c10d \
    --node_rank=${NODE_RANK}\
    --rdzv_conf 'read_timeout=420' \
    --nnodes="${WORLD_SIZE}" \
    --rdzv_id 12349 \
    --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
    --master_addr ${MASTER_ADDR} \
    scripts/train.py \
    configs/astro-cpt/OLMo-1B.yaml \
    --reset_trainer_state \
    --remote_save_folder=null \
    --save_overwrite \
    --reset_optimizer_state \
    --data.paths=[${data_folder}/${filter_strategy}/train/part-0-00000.npy] \
    --load_path=${path_to_checkpoint} \
    --save_folder=${save_root}/${filter_strategy}/OLMo-1B-final \
    --restart_from_unsharded_checkpoint=True \
    --filter_strategy=${filter_strategy} \
    --scheduler.t_warmup=${t_warmup} \