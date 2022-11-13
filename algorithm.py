from typing import Callable, List

from hiper_params import (
    POPULATION_SIZE,
    CROSS_PROBABILITY,
    MUTATION_STRENGTH,
    ITERATIONS,
)
from utils import (
    cost_function,
    find_best_individual,
    one_point_crossover,
    gauss_mutation,
    init_population,
    cube_constrains,
    roulette_reproduction
)


def evolution(
        cost_function: Callable,
        reproduction_function: Callable,
        population_members: List,
        mutation_strength: float,
        cross_probability: float,
        iterations: int,
):
    population_grades = [cost_function(individual) for individual in population_members]
    best_individual, best_grade = find_best_individual(
        population_members, population_grades
    )
    best_in_interation = []
    for i in range(iterations):
        # reproduction
        mutant_population = reproduction_function(population_members, population_grades)

        # genetic operations
        mutant_population = one_point_crossover(mutant_population, cross_probability)
        mutant_population = gauss_mutation(mutant_population, mutation_strength)

        # grading
        mutants_grades = [cost_function(individual) for individual in mutant_population]
        best_mutant, best_mutant_grade = find_best_individual(
            mutant_population, mutants_grades
        )
        if best_mutant_grade <= best_grade:
            best_grade = best_mutant_grade
            best_individual = best_mutant
        best_in_interation.append(best_grade)
        # generational succession
        population_members = mutant_population
        population_grades = mutants_grades

    return best_individual, best_grade, best_in_interation


if __name__ == "__main__":
    population = init_population(cube_constrains, POPULATION_SIZE)
    point, grade, bests_in_pop = evolution(
        cost_function, roulette_reproduction, population, MUTATION_STRENGTH, CROSS_PROBABILITY, ITERATIONS
    )
    print(bests_in_pop[:4])
    print(bests_in_pop[-4:])
