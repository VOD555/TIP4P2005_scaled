module load gromacs/2021.1/cuda11.1/gcc-7.5.0/avx/bare/


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