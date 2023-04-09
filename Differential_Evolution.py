import random
import numpy as np

# Define the mutation function
import random

def differential_evolution(objective_func, bounds, pop_size=10, mutation=0.8, crossover=0.7, max_iter=1000, tol=1e-6):
    """
    Differential evolution algorithm for optimizing a function with N parameters.

    Args:
        objective_func (function): Objective function to be optimized.
        bounds (list): List of tuples specifying the upper and lower bounds of each parameter.
        pop_size (int): Population size.
        mutation (float): Mutation rate.
        crossover (float): Crossover rate.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.

    Returns:
        tuple: Tuple containing the best solution found and its corresponding objective value.
    """

    # Initialize population
    pop = [[random.uniform(b[0], b[1]) for b in bounds] for i in range(pop_size)]

    # Main loop
    for i in range(max_iter):

        # Evaluate objective function for each individual
        obj_vals = [objective_func(p) for p in pop]

        # Find best individual in population
        best_idx = min(range(pop_size), key=lambda i: obj_vals[i])

        # Check for convergence
        if obj_vals[best_idx] < tol:
            break

        # Generate new population
        new_pop = []
        for j in range(pop_size):

            # Select three random individuals from population
            idxs = random.sample(range(pop_size), 3)
            a, b, c = pop[idxs[0]], pop[idxs[1]], pop[idxs[2]]

            # Perform mutation
            mutant = mutation_func(a, b, c, mutation, bounds)

            # Perform crossover
            trial = crossover_func(mutant, pop[j], crossover)

            # Evaluate trial individual
            trial_obj_val = objective_func(trial)

            # Select individual for next generation
            if trial_obj_val < obj_vals[j]:
                new_pop.append(trial)
            else:
                new_pop.append(pop[j])

        pop = new_pop

    # Find best individual in final population
    obj_vals = [objective_func(p) for p in pop]
    best_idx = min(range(pop_size), key=lambda i: obj_vals[i])
    best_sol = pop[best_idx]
    best_obj_val = obj_vals[best_idx]

    return (best_sol, best_obj_val)


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