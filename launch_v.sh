#!/bin/bash
# Node resource configurations
#SBATCH --job-name=train_jamot
#SBATCH --mem=16G
#SBATCH --cpus-per-gpu=4

# for normal t4v2,t4v1,a40
# for high t4v2
# for deadline t4v2,t4v1,a40
#SBATCH --partition=a40

#SBATCH --gres=gpu:1
#SBATCH --account=deadline
#SBATCH --qos=deadline
#SBATCH --output=./logs/slurm-%j.out
#SBATCH --error=./logs/slurm-%j.err

# Append is important because otherwise preemption resets the file
#SBATCH --open-mode=append

echo `date`: Job $SLURM_JOB_ID is allocated resource

# creating dirs
ln -sfn /checkpoint/${USER}/${SLURM_JOB_ID} $PWD/checkpoint/${SLURM_JOB_ID}
touch /checkpoint/${USER}/${SLURM_JOB_ID}/DELAYPURGE
mkdir /checkpoint/${USER}/${SLURM_JOB_ID}/checkpoints

source /ssd003/home/${USER}/.bashrc
source /ssd003/home/${USER}/venvs/jax-env/bin/activate


python main.py --config configs/embrio/phot.py \
               --workdir $PWD/checkpoint/${SLURM_JOB_ID} \
               --mode '5seeds'

echo `date`: "Job $SLURM_JOB_ID finished running, exit code: $?"

date=$(date '+%Y-%m-%d')
archive=$HOME/finished_jobs/$date/$SLURM_JOB_ID
mkdir -p $archive

cp ./logs/slurm-$SLURM_JOB_ID.out $archive/job.out
cp ./logs/slurm-$SLURM_JOB_ID.err $archive/job.err
