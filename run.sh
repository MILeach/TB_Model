#!/bin/bash
#SBATCH --account=dcs-collab
#SBATCH --partition=dcs-gpu-test 
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=1
#SBATCH --time=0-00:15
#SBATCH --job-name=gpurun

module load CUDA/10.0.130

cd ~/TB_Model/FLAMEGPU/examples/TB_Model

for f in $1/*.xml
do
  echo "Running file $f"
  ./bin/linux-x64/Release_Console/Project $f $2 XML_output_frequency 0
  outputFileName=$(basename $f .xml)
  echo "Copying output to $outputFileName"
  cp iterations/person-output-0.csv iterations/output/$outputFileName.csv
done
