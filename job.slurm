#!/bin/sh

#SBATCH -J sectrl
#SBATCH --output=slurm/logs/sectrl.%j
#SBATCH --error=slurm/errors/sectrl.%j
#
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --time 0-00:10:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=1


SCRIPT="/dss/dsshome1/07/go69lir/SECTRL/src/main.py --benchmark d8"
CODE_DIR="/dss/dsshome1/07/go69lir/SECTRL"

srun --cpu-bind=verbose,socket \
     bash -c "cd ${CODE_DIR} && source .venv/bin/activate && python -u ${SCRIPT}"