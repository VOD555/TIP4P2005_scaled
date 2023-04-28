import numpy as np
import random
from os import path
from subprocess import call
from objective_func import objective
import logging
import pandas as pd
from Differential_Evolution import mutation_func, crossover_func
from production import check_job_status, production_simulation

if __name__ == '__main__':
    # range of the parameters, latter defined in a yaml file
    # Initial settings.
    pop_size = 20
    max_iter=100
    charges = (0.9, 1.2)
    sigma = (0.29, 0.33)
    epsilon = (0.5, 1.0)
    OD = (0., 0.03/(np.cos(np.radians(52.26))* 0.09572)/2)
    bounds = [charges, sigma, epsilon, OD]

    temp = '/nfs/homes4/sfan/Projects/Methods/TIP4P2005_scaled/sim_template'
    dir = '/nfs/homes4/sfan/Projects/TIP4P/OD'

    df = pd.read_csv('/nfs/homes4/sfan/Projects/Methods/TIP4P2005_scaled/rdf.csv')                                                 
    rdfrdf = df.OO                                                              
    ref = [np.array([45, 0.99565, 0.797, 2.3]), rdfrdf] 

    logger = logging.getLogger(path.join(dir, 'logging.log'))

    # Generate initial solutions
    logger.info('Generating inital solutions.')

    solutions = [[random.uniform(b[0], b[1]) for b in bounds] for i in range(pop_size)]
    # solutions = np.load('/nfs/homes4/sfan/Projects/TIP4P/refine0/solutions.npy')
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
    account = "TIP4P"
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

    best_idx = np.argmin(valid_samples[:,-1])
    best_solution = valid_samples[best_idx, :3]
    print('Best sample gives an objective function of {}'.format(valid_samples[best_idx, -1]))
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
                mutant = mutation_func(a, b, c, best_solution, mutation, bounds, 'current_to_best')

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
        best_solution = valid_samples[best_idx, :3]
        bestobj = valid_samples[best_idx, -1]
        print('Best solution is {0}, {1}, {2}, and objective function is {3}'.format(
                    best_solution[0], best_solution[1], best_solution[2], bestobj))
        
        if bestobj < 0.01:
            print('Converged.')
            break
    else:
        print("Didn't get converged results after {} rounds of DE.".format(max_iter))


            