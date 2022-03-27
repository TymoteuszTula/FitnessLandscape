#!/bin/bash

#SBATCH --job-name=RandomSamplesCorrConn
#SBATCH --output=/home/home/tt343/CorrelationConnection/run/output/icarus/outputs/array_%A_%a.out
#SBATCH --error=/home/home/tt343/CorrelationConnection/run/output/icarus/errors/array_%A_%a.err
#SBATCH --array=1-16%4
#SBATCH --mail-type=END
#SBATCH --mail-user=tt343@kent.ac.uk 
#SBATCH --cpus-per-task=4
echo "Starting job $SBATCH_JOB_NAME $SLURM_JOB_ID task ${SLURM_ARRAY_TASK_ID} running on $SLURM_JOB_NODELIST"
python /home/home/tt343/CorrelationConnection/run/run_on_icarus.py -i ${SLURM_ARRAY_TASK_ID}
echo "Finished job $SLURM_JOB_ID on $HOSTNAME node list $SLURM_JOB_NODELIST"
