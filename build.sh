#!/bin/bash
#SBATCH --account=dcs-res 
#SBATCH --partition=dcs-gpu-test 
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=1
#SBATCH --time=0-00:10
#SBATCH --job-name=gpubuildtest

module load CUDA/10.0.130

cd FLAMEGPU/examples/TB_Model
make console SMS=70
~                        
