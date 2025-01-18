#!/bin/bash
#SBATCH -J sft-dscoder-wd0.-norm1.0          # Job name
#SBATCH -o slurm-outputs/%x.o%j       # Name of stdout output file
#SBATCH -e slurm-outputs/%x.e%j       # Name of stderr output file
#SBATCH -p gh          # Queue (partition) name
#SBATCH -N 1              # Total # of nodes
##SBATCH --ntasks-per-node=1 
#SBATCH -t 2:30:00        # Run time (hh:mm:ss)
#SBATCH -A AST24021       # Allocation name (req'd if you have more than 1)

export CUDA_VISIBLE_DEVICES=0

export WANDB_MODE=offline

gpu_count=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
bs=32
per_device_train_batch_size=32
grad_acc=$((bs / gpu_count / per_device_train_batch_size))

weight_decay=0.1
max_grad_norm=1.0

seed=42
warmup_ratio=0.03
max_seq_length=2048

for lr in 5e-5 1e-4 2e-4 5e-4 1e-3
do

output_dir=${SCRATCH}/sft/dscode-1B-lr${lr}-wd${weight_decay}-warmup${warmup_ratio}-norm${max_grad_norm}-len${max_seq_length}-seed${seed}
# model_name_or_path=${SCRATCH}/base_models/deepseek/hf/deepseek-coder-1.3b-base

accelerate launch --config_file="fsdp_config.yaml" \
    --main_process_port 29600 \
    sft_train.py \
    --output_dir="${output_dir}" \
    --seed=${seed} \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --per_device_train_batch_size=${per_device_train_batch_size} \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=${grad_acc} \
    --max_seq_length=${max_seq_length} \
    --max_grad_norm=${max_grad_norm} \
    --optim="adamw_torch" \
    --dataset_text_field="sft_text" \
    --lr_scheduler_type="linear" \
    --warmup_ratio=${warmup_ratio} \
    --eval_strategy="epoch" \
    --save_strategy="epoch" \
    --save_total_limit=2 \
    --load_best_model_at_end=True \
    --logging_strategy="steps" \
    --logging_first_step=True \
    --logging_steps=5 \
    --eval_on_start=True \
    --report_to="wandb" \
    --run_name="sft-dscoder-1B" \

done