import numpy as np
import random
from os import path
import subprocess
import time
from subprocess import call
import shutil
from objective_func import objective
import logging

def mutation_func(a, b, c, mutation, bounds):
    """
    Mutation function for differential evolution algorithm.

    Args:
        a (list): First parent.
        b (list): Second parent.
        c (list): Third parent.
        mutation (float): Mutation rate.
        bounds (list): List of tuples specifying the upper and lower bounds of each parameter.

    Returns:
        list: Mutated individual.
    """

    mutant = [a[i] + mutation * (b[i] - c[i]) for i in range(len(a))]
    mutant = [min(max(mutant[i], bounds[i][0]), bounds[i][1]) for i in range(len(mutant))]

    return mutant


def crossover_func(mutant, parent, crossover):
    """
    Crossover function for differential evolution algorithm.

    Args:
        mutant (list): Mutated individual.
        parent (list): Parent individual.
        crossover (float): Crossover rate.

    Returns:
        list: Trial individual.
    """

    trial = [mutant[i] if random.random() < crossover else parent[i] for i in range(len(mutant))]

    return trial

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

    with open(topol, 'w') as ff:
        ff.writelines( data )
    s = path.join(dir, 'submit.sh')
    call(s)

def check_job_status(account):
    while True:
        output = subprocess.check_output(["squeue", "-u", account])
        lines = output.decode().split("\n")
        num_jobs = len(lines) - 2  # subtract two for header and empty line
        if num_jobs == 0:
            break
        time.sleep(150)  # wait for 10 seconds before checking again

    print("All jobs under account", account, "have finished.")

if __name__ == '__main__':
    # range of the parameters, latter defined in a yaml file
    # Initial settings.
    pop_size = 20
    max_iter=100
    charges = (0.5*1.1128, 1.1*1.1128)
    sigma = (0.5*0.31589, 1.2*0.31589)
    epsilon = (0.5*0.77490, 1.2*0.77490)
    bounds = [charges, sigma, epsilon]

    temp = '/nfs/homes4/sfan/Projects/Methods/TIP4P2005_scaled/sim_template'
    dir = '/nfs/homes4/sfan/Projects/TIP4P/100round'
    ref = np.array([45, 0.99565, 0.0007972])

    logger = logging.getLogger(path.join(dir, 'logging.log'))

    # Generate initial solutions
    logger.info('Generating inital solutions.')

    solutions = [[1.1128, 0.31589, 0.77490]] + [[random.uniform(b[0], b[1]) for b in bounds] for i in range(pop_size-1)]
    solutions_array = np.array(solutions)

    logger.info('Save solutions to {}.'.format(path.join(dir, 'solutions')))

    np.save(path.join(dir, 'solutions'), solutions_array)
    n=0

    # Start running simulation
    logger.info('Submitting simulation jobs.')
    for solution in solutions:
        production_simulation(solution, temp, dir, n)
        n += 1

    logger.info('Waiting simulations.')
    account = "sfan"
    check_job_status(account)

    # Predict physical properties and objective function
    logger.info('Collecting results.')
    res = []
    for i in range(pop_size):
        res.append(objective(path.join(dir, str(i)), ref))
    res = np.array(res)
    np.save(path.join(dir, 'res'), res)

    samples = np.hstack((solutions_array, res))
    np.save(path.join(dir, 'samples'), samples)

    valid_samples = samples[samples[:,3]!=0]
    valid_solutions = valid_samples[:,:3].tolist()

    best_idx = np.argmax(valid_samples[:,-1])
    best_solution = valid_samples[best_idx, :]
    logger.info('Best solution is {0}, {1}, {2}, and objective function is'.format(
        best_solution[0], best_solution[1], best_solution[2], best_solution[-1]))

    mutation = 0.8
    crossover = 0.7

    # Starting sampling new solutions using differential evolution.
    logger.info('Generating new solutions via DE.')

    for iter in range(max_iter):
        new_solutions = solutions.copy()
        for i in range(pop_size):
            if res[i,0] == 0:
                new_solutions[i] = [random.uniform(b[0], b[1]) for b in bounds]
            else:
                idxs = random.sample(range(len(valid_solutions)), 3)
                a, b, c = valid_solutions[idxs[0]], valid_solutions[idxs[1]], valid_solutions[idxs[2]]

                # Perform mutation
                mutant = mutation_func(a, b, c, mutation, bounds)

                # Perform crossover
                trial = crossover_func(mutant, solutions[i], crossover)
                new_solutions[i] = trial
        new_solutions_array = np.array(new_solutions)
        np.save(path.join(dir, 'new_solution'), new_solutions_array)

        # Perfrom simulations
        for solution in new_solutions:
            production_simulation(solution, temp, dir, n)
            n += 1

        check_job_status(account)

        # Evaluate solutions
        new_res = []
        for i in range(n-pop_size, n):
            new_res.append(objective(path.join(dir, str(i)), ref))
        new_res = np.array(new_res)
        np.save(path.join(dir, 'new_res'), new_res)

        new_samples = np.hstack((new_solutions_array, new_res))
        samples = np.vstack((samples, new_samples))
        np.save(path.join(dir, 'samples'), samples)

        # Detemine the new solutions for next round of differential evolution

        for i in range(pop_size):
            if res[i,0] == 0:
                solutions[i] = new_solutions[i]
                res[i, :] = new_res[i, :]
            elif new_res[i,0] != 0:
                if res[i, -1] > new_res[i, -1]:
                    solutions[i] = new_solutions[i]
                    res[i, :] = new_res[i, :]
        np.save(path.join(dir, 'solutions'), np.array(solutions))
        np.save(path.join(dir, 'res'), np.array(res))

        intermediate_samples = np.hstack((solutions, res))
        valid_samples = intermediate_samples[intermediate_samples[:,3]!=0]
        valid_solutions = valid_samples[:,:3].tolist()
    
        best_idx = np.argmax(valid_samples[:,-1])
        best_solution = valid_samples[best_idx, :]
        logger.info('Best solution is {0}, {1}, {2}, and objective function is'.format(
                    best_solution[0], best_solution[1], best_solution[2], best_solution[-1]))
        if best_solution[-1] < 0.01:
            print('Converged.')
            break
    else:
        print("Didn't get converged results after {} rounds of DE.".format(max_iter))


            