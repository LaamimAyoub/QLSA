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
export nb_proc=60
export EPSILON=0.1
cd /home/$USER/QLSA
conda run -n qlsa python  QLSA5alg2_ttq_last.py


export CONDA_ENVS_PATH=/home/$USER/envs
module load Anaconda3
source activate qlsa
unset PYTHONPATH
echo $CONDA_PREFIX
export nb_proc=60
export EPSILON=0.2
cd /home/$USER/QLSA
conda run -n qlsa python  QLSA5alg2_ttq_last.py


export CONDA_ENVS_PATH=/home/$USER/envs
module load Anaconda3
source activate qlsa
unset PYTHONPATH
echo $CONDA_PREFIX
export nb_proc=60
export EPSILON=0.3
cd /home/$USER/QLSA
conda run -n qlsa python  QLSA5alg2_ttq_last.py




export CONDA_ENVS_PATH=/home/$USER/envs
module load Anaconda3
source activate qlsa
unset PYTHONPATH
echo $CONDA_PREFIX
export nb_proc=60
export EPSILON=0.5
cd /home/$USER/QLSA
conda run -n qlsa python  QLSA5alg2_ttq_last.py




export CONDA_ENVS_PATH=/home/$USER/envs
module load Anaconda3
source activate qlsa
unset PYTHONPATH
echo $CONDA_PREFIX
export nb_proc=60
export EPSILON=0.7
cd /home/$USER/QLSA
conda run -n qlsa python  QLSA5alg2_ttq_last.py




export CONDA_ENVS_PATH=/home/$USER/envs
module load Anaconda3
source activate qlsa
unset PYTHONPATH
echo $CONDA_PREFIX
export nb_proc=60
export EPSILON=0.9
cd /home/$USER/QLSA
conda run -n qlsa python  QLSA5alg2_ttq_last.py


export CONDA_ENVS_PATH=/home/$USER/envs
module load Anaconda3
source activate qlsa
unset PYTHONPATH
echo $CONDA_PREFIX
export nb_proc=60
export EPSILON=1.0
cd /home/$USER/QLSA
conda run -n qlsa python  QLSA5alg2_ttq_last.py