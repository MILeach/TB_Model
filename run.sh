#!/bin/bash
#SBATCH --account=dcs-res
#SBATCH --partition=dcs-gpu-test 
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=1
#SBATCH --time=0-00:15
#SBATCH --job-name=gpurun

module load CUDA/10.0.130

cd ~/TB_Model/FLAMEGPU/examples/TB_Model

FILES = $1/*.xml
for f in $FILES 
do
  echo "Running file $f"
  ./bin/linux-x64/Release_Console/Project $1 $2 XML_output_frequency 0
done
