from typing import Callable, Sequence
from utils import cost_function, find_best_individual, roulette_reproduction, one_point_crossover, gauss_mutation


def evolution(
    cost_function: Callable,
    population_members: Sequence,
    mutation_strength: float,
    cross_probability: float,
):
    population_grades = [cost_function(individual) for individual in population_members]
    best_individual, best_grade = find_best_individual(population_members, population_grades)
    for _ in range(1000):
        # reproduction
        mutant_population = roulette_reproduction(population_members, population_grades)

        # genetic operations
        mutant_population = one_point_crossover(mutant_population, cross_probability)
        mutant_population = gauss_mutation(mutant_population, mutation_strength)

        # grading
        mutants_grades = [cost_function(individual) for individual in mutant_population]
        best_mutant, best_mutant_grade = find_best_individual(mutant_population, mutants_grades)
        if best_mutant_grade <= best_grade:
            best_grade = best_mutant_grade
            best_individual = best_mutant

        # generational succession
        population_members = mutant_population
        population_grades = mutants_grades
        
    return best_individual, best_grade

if __name__ == "__main__":
    print(cost_function((0, 0)))
    print(cost_function((1, 0)))
    print(cost_function((10, 0)))
