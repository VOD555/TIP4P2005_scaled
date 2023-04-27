#!/bin/bash
 
#SBATCH -J TIP4P                 # name
#SBATCH --exclude=tubiwan
#SBATCH -N 1                           # 1 node
#SBATCH -n 16                           # 6 cores in total
#SBATCH --gres=gpu:1               # Send a notification when the job starts, stops, or fails
#SBATCH --chdir=.                      # start from submission directory
#SBATCH -t 2-24:00                     # wall time D-HH:MM
 
# run Gromacs 2021.1
 
# threaded version (fastest on a single node)

gmx grompp -f step4.0_minimization.mdp -o step4.0_minimization.tpr -c step3_input.pdb -r step3_input.pdb -p topol.top -n index.ndx -maxwarn -1
gmx mdrun -v -deffnm step4.0_minimization


# Equilibration
gmx grompp -f step4.1_equilibration.mdp -o step4.1_equilibration.tpr -c step4.0_minimization.gro -r step3_input.pdb -p topol.top -n index.ndx
gmx mdrun -v -deffnm step4.1_equilibration


# Production
set pstep = step4.0_minimization
gmx grompp -f step5_production.mdp -o step5.tpr -c step4.1_equilibration.gro -p topol.top -n index.ndx

gmx mdrun -v -deffnm step5

echo "0" | gmx trjconv -f step5.xtc -s step5.tpr -ur compact -pbc mol -o md.xtc 

# 1ns NVT
for i in {0..4}
do
    cd md$i
    gmx grompp -f ../nvt.mdp -c ../step5.gro -p ../topol.top -n ../index.ndx -o nvt.tpr
    gmx mdrun -deffnm nvt -v
    echo "0" | gmx trjconv -f nvt.xtc -s nvt.tpr -ur compact -pbc mol -o nvtc.xtc 
    result=$(awk 'END{print $1*$2*$3}' nvt.gro)

    echo $result | gmx energy -f nvt.edr -s nvt.tpr -evisco -vis
    
    echo "3" | gmx msd -f nvt.xtc -s nvt.tpr -n ../ow.ndx

    rm energy.xvg
    cd ..
done

source ~/miniconda3/etc/profile.d/conda.sh
conda activate py39

python get_rdf.py