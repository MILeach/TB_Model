#!/bin/bash
#SBATCH --account=dcs-res
#SBATCH --partition=dcs-gpu-test 
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=1
#SBATCH --time=0-00:15
#SBATCH --job-name=gpurun

module load CUDA/10.0.130

cd ~/TB_Model/

for f in input/$1/*.xml
do
  echo "Running file $f"
  ./FLAMEGPU/examples/TB_Model/bin/linux-x64/Release_Console/Project $f $2 XML_output_frequency 0
  outputFileName=$(basename $f .xml)
  echo "Copying output to $outputFileName"
  mkdir output
  cd output
  mkdir $1
  cd ..
  cp input/$1/person-output-0.csv output/$1/$outputFileName.csv
done
cd input/$1
rm person-output-0.csv
