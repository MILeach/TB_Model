#!/bin/bash
#SBATCH --account=dcs-collab
#SBATCH --partition=dcs-gpu-test 
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=1
#SBATCH --time=0-00:15
#SBATCH --job-name=gpubuild

module load CUDA/10.0.130

cd ~/TB_Model/FLAMEGPU/examples/TB_Model
make console SMS=60
