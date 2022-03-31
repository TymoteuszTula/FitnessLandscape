#!/bin/bash
#SBATCH --job-name=RandomSamplesCorrConn
#SBATCH --output=/home/tt343/CorrelationConnection/run/output/icarus/outputs/run31032022/array_%A_%a.out
#SBATCH --error=/home/tt343/CorrelationConnection/run/output/icarus/errors/run31032022/array_%A_%a.err
#SBATCH --array=1-90%4
#SBATCH --mail-type=END
#SBATCH --mail-user=tt343@kent.ac.uk 
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1000
#SBATCH --partition=cpu

echo "Starting job $SBATCH_JOB_NAME $SLURM_JOB_ID task ${SLURM_ARRAY_TASK_ID} running on $SLURM_JOB_NODELIST"
python /home/tt343/CorrelationConnection/run/run_on_icarus.py ${SLURM_ARRAY_TASK_ID} --input_folder run31032022/
echo "Finished job $SLURM_JOB_ID on $HOSTNAME node list $SLURM_JOB_NODELIST"
