#!/bin/bash
#SBATCH -J olmo-10k           # Job name
#SBATCH -o slurm-outputs/olmo-10k.o%j       # Name of stdout output file
#SBATCH -e slurm-outputs/olmo-10k.e%j       # Name of stderr output file
#SBATCH -p gh          # Queue (partition) name
#SBATCH -N 2
#SBATCH -n 2              # Total # of nodes
#SBATCH -t 01:00:00        # Run time (hh:mm:ss)
#SBATCH -A AST24021       # Allocation name (req'd if you have more than 1)

srun /work/07144/yw23374/vista/OLMo/slurm-scripts/run_cpt_7B_multi_node.sh $SLURM_JOB_NODELIST $SLURM_JOB_NUM_NODES
