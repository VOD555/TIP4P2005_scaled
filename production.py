import shutil
from os import path
from subprocess import call
import subprocess
import time

def production_simulation(solution, temp_dir, repository, n):
    """
    Submit slurm job to perform simulations based on the given parameters in solution.

    Args:
        solution (list): list of parameters.
        temp_dir (string): path to the simulation template directory.
        repository (string): path to the output directory.
        n (int): number of the current list of parameters.

    """

    dir = path.join(repository, str(n))
    shutil.copytree(temp_dir, dir)
    topol = path.join(dir, 'topol.top')
    with open(topol, 'r') as ff:
        data = ff.readlines()
    data[6] = 'IW      0           -{0:.4f}      D   0.0           0.0\n'.format(round(solution[0], 4))
    data[7] = 'OWT4    15.9994      0.0000      A   {0:.5f}       {1:.5f}\n'.format(solution[1], solution[2])
    data[8] = 'HW      1.0079       {0:.5f}     A   0.00000E+00   0.00000E+00\n'.format(round(solution[0], 4)/2)
    data[53] = '4       1       2       3       1       {0:.8f}      {0:.8f} \n'.format(round(solution[3], 8))

    with open(topol, 'w') as ff:
        ff.writelines( data )
    s = path.join(dir, 'submit.sh')
    call(s)

def check_job_status(account):
    while True:
        output = subprocess.check_output(["squeue", "--name", account])
        lines = output.decode().split("\n")
        num_jobs = len(lines) - 2  # subtract two for header and empty line
        if num_jobs == 0:
            break
        time.sleep(150)  # wait for 10 seconds before checking again

    print("All jobs under account", account, "have finished.")