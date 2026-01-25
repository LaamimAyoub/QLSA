#!/bin/bash
#SBATCH --job-name=NohaJob
#SBATCH --partition=mediumq  
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

export CONDA_ENVS_PATH=/home/$USER/envs
module load Anaconda3 
source activate qlsa
unset PYTHONPATH
echo $CONDA_PREFIX
export nb_proc=50
cd /home/$USER/QLSA
conda run -n qlsa python  QLSA5alg2_ttq_last.py


