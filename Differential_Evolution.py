import random
import numpy as np

# Define the mutation function
import random

def mutation_func(a, b, c, best, mutation, bounds, method='rand1'):
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
    if method == 'rand1':
        mutant = [a[i] + mutation * (b[i] - c[i]) for i in range(len(a))]
    elif method == 'current_to_best':
        mutant = [a[i] + mutation * (best[i] - a[i]) + mutation * (b[i] - c[i])]

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