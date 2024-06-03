#!/bin/bash
#SBATCH --job-name=two_rooms_fifty_fifty_goal_64_train_ratio
#SBATCH --output=out/two_rooms_fifty_fifty_goal_64_train_ratio_%A_%a.out
#SBATCH --error=err/two_rooms_fifty_fifty_goal_64_train_ratio_%A_%a.err
#SBATCH --array=0-9
#SBATCH --time=2:59:59
#SBATCH --mem=10G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=rrg-mbowling-ad
#SBATCH --mail-user=kapeluck@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

# SETUP
module load StdEnv/2020
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load python/3.9
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements_cc.txt
pip install --no-index matplotlib # needed for Two Rooms
module load gcc/9.3.0 cuda/11.8 cudnn/8.6
export LD_LIBRARY_PATH="$CUDA_HOME/lib64;$EBROOTCUDNN/lib"

mkdir -p ./logs/$SLURM_JOB_NAME

python example.py \
    --logdir="./logs/${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}" \
    --seed="${SLURM_ARRAY_TASK_ID}"

touch ./logs/$SLURM_JOB_NAME/$SLURM_ARRAY_TASK_ID/complete
