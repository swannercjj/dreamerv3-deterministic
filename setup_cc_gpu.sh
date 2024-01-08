#!/bin/bash
module load StdEnv/2020
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load python/3.9
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
# pip install pysocks
pip install --no-index --upgrade pip
pip install --no-index -r requirements_cc.txt
# Need to load cuda after everything else for some reason
module load gcc/9.3.0 cuda/11.8 cudnn/8.6
export LD_LIBRARY_PATH="$CUDA_HOME/lib64;$EBROOTCUDNN/lib"