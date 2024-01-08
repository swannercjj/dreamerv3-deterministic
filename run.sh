#!/bin/bash
#SBATCH --job-name=pendulum
#SBATCH --output=out/pendulum_%A_%a.out
#SBATCH --error=err/pendulum_%A_%a.err
#SBATCH --array=0-20
#SBATCH --time=48:00:00
#SBATCH --mem=6G
#SBATCH --account=rrg-mbowling-ad
#SBATCH --mail-user=kapeluck@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

module load python/3.9
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements_cc.txt
# Need to load cuda after everything else for some reason
module load gcc/9.3.0 cuda/11.8 cudnn/8.6
export LD_LIBRARY_PATH="$CUDA_HOME/lib64;$EBROOTCUDNN/lib"

python example.py \
    --logdir="./logs/${SLURM_ARRAY_TASK_ID}" \
    --seed="${SLURM_ARRAY_TASK_ID}"

touch ./logs/$SLURM_ARRAY_TASK_ID/complete