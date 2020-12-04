## Build Instructions

### Building on Bessemer HPC

1. Connect to the Bessemer HPC system and login
2. Run the command `git clone https://github.com/MILeach/TB_Model.git` to copy the project files into your user area.
3. Run `cd TB_Model` to move into the project directory
4. Run `sbatch build.sh` to submit an HPC job which will build the project

### Building locally on linux

Requirements for building locally
- Nvidia graphics card
- A valid CUDA installation

Build Instructions

1. Open a terminal and navigate to the folder where you wish to set up the project
2. Run the command `git clone https://github.com/MILeach/TB_Model.git` to copy the project files into your user area.
3. Run `cd TB_Model/FLAMEGPU/examples/TB_Model` to move into the project directory
4. Run `make` to build the project

## Setting up the Model
- Navigate to the `TB_Model/FLAMEGPU/examples/TB_Model/initializer` folder
- Modify histo.csv to adjust the number of people in each sex/age category
- run `python3 preprocess.py data.in` to generate the data.in file which describes the population
- For an HPC build, copy the created `data.in` file to the `TB_Model/input` folder. For a local build, copy the file to `TB_Model/FLAMEGPU/examples/TB_Model/iterations` folder.
- For an HPC build, create a folder inside the `TB_Model/input` folder and copy the `network.json` file into it. 
- Parameters are controlled using the FLAMEGPU XML input files. For an HPC build, place any FLAMEGPU XML input files you wish to run together in a folder in the `TB_Model/input` directory. For a local build, place your FLAMEGPU XML input file in the `TB_Model/FLAMEGPU/examples/TB_Model/iterations` folder.

## Running a Simulation

### Running a simulation on Bessemer HPC

1. Navigate to the top level `TB_Model` directory
2. Ensure the `input` folder contains the correct `data.in` file and a folder containing a copy of `network.json` and the XML input files you wish to run as described in the *Setting up the Model* section
3. Run the command `sbatch run.sh input_folder_name number_of_iterations` where `input_folder_name` is the name of the folder that contains the XML files and `number_of_iterations` is the number of timesteps you wish to run the simulation for
4. When the job is completed, the output files should be available in the `output` folder

### Running a simulation locally on linux (single XML file only)

1. Navigate to the `TB_Model/FLAMEGPU/examples/TB_Model` directory
2. Ensure the XML input file, `network.json` file and `data.in` file are present in the iterations folder
3. Run the command `./bin/linux-x64/Release_Console/Project iterations/input_filename.xml number_of_iterations XML_output_frequency 0`

## Other Bessemer HPC utilities
- `sacct -v` - lists your active jobs
- `scancel job_number` - cancels the job with id job_number 