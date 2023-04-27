#!/bin/bash
 
#SBATCH -J ECC                 # name
#SBATCH -N 1                           # 1 node
#SBATCH -n 1                           # 6 cores in total
#SBATCH --chdir=.                      # start from submission directory
 
# run Gromacs 2021.1
 
# threaded version (fastest on a single node)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate py39
cd /nfs/homes4/sfan/Projects/Methods/TIP4P2005_scaled/
python main.py

