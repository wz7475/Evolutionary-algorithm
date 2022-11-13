import copy
from typing import Tuple, List
from math import e as e_constant
from random import uniform, choices, randint
import numpy as np


def _manhattandistance(x: Tuple, y: Tuple) -> float:
    return abs(x[0] - y[0]) + abs(x[1] - y[1])


def cost_function(localisation: Tuple) -> float:
    resources = {
        "z1": [20, [1, 1]],
        "z2": [10, [-0.5, 1]],
        "z3": [5, [-1, -0.5]],
        "z4": [10, [1, -1]]
    }
    total_cost = 0
    for resource_name in resources:
        quantity, resource_localisation = resources[resource_name]
        unit_cost = 1 - pow(
            e_constant, (-1) * _manhattandistance(resource_localisation, localisation)
        )
        total_cost += quantity * unit_cost
    return total_cost


def _get_cube_constrains(resources: dict) -> dict:
    x_min, x_max, y_min, y_max = None, None, None, None
    for resource_name in resources:
        _, resource_localisation = resources[resource_name]
        x, y = resource_localisation
        if x_min is None:
            x_min = x
        elif x_min > x:
            x_min = x
        if y_min is None:
            y_min = y
        elif y_min > y:
            y_min = y
        if x_max is None:
            x_max = x
        elif x_max < x:
            x_max = x
        if y_max is None:
            y_max = y
        elif y_max < y:
            y_max = y
    return {
        "x": [x_min, x_max],
        "y": [y_min, y_max],
    }


def init_population(constrains: dict, quantity: int):
    return [
        (
            uniform(constrains["x"][0], constrains["x"][1]),
            uniform(constrains["y"][0], constrains["y"][1]),
        )
        for _ in range(quantity)
    ]


def roulette_reproduction(population: List, population_grades: List):
    max_grade = max(population_grades)
    min_grade = min(population_grades)
    """
    basic proportion: (1 - grade / sum_of_grades) does not distinguish enough week and strong individuals,
    in order to stress difference min max scaler is applied,
    it may occur that some individuals probability is equal to 0 or 1, edge vales are 'soften'  
    """

    def pseudo_min_max_scaler_modified(grade):
        nominator = grade - min_grade
        denominator = max_grade - min_grade
        scaled = 1 - nominator / denominator
        if scaled <= 0.1:
            scaled += 0.02
        elif scaled >= 0.9:
            scaled -= 0.02
        return scaled

    distribution = [
        pseudo_min_max_scaler_modified(grade) for grade in population_grades
    ]
    """take k random choices based on weights - probability distribution"""
    return choices(population=population, weights=distribution, k=len(population))


def roulette_reproduction_weak(population: List, population_grades: List):
    distribution = [1 - grade / sum(population_grades) for grade in population_grades]
    return choices(population=population, weights=distribution, k=len(population))


def find_best_individual(
        population: List, population_grades: List
) -> Tuple[float, float]:
    min_index = 0
    min_grade = population_grades[min_index]
    for index in range(1, len(population_grades)):
        if population_grades[index] <= population_grades[index]:
            min_index = index
            min_grade = population_grades[index]
    return population[min_index], min_grade


def one_point_crossover(population: List, crossover_probability: float) -> List:
    new_population = []
    for_crossover = []
    for individual in population:
        if uniform(0, 1) >= crossover_probability:
            for_crossover.append(individual)
        else:
            new_population.append(individual)
    for_crossover_length = len(for_crossover)
    if for_crossover_length % 2 != 0:
        new_population.append(for_crossover.pop(randint(0, for_crossover_length - 1)))
    for i in range(len(for_crossover) // 2):
        first_parent = population[i]
        second_parent = population[i + 1]
        first_child = [first_parent[0], second_parent[1]]
        second_child = [second_parent[0], first_child[1]]
        new_population.append(first_child)
        new_population.append(second_child)
    return new_population


def gauss_mutation(population: np.array, mutation_strength: float) -> np.array:
    population = np.array(population)
    normal_distribution = np.random.normal(0, 1, len(population))
    mutant_population = copy.deepcopy(population)
    for i in range(len(population)):
        x = mutant_population[i][0] * mutation_strength * normal_distribution[i]
        y = mutant_population[i][1] * mutation_strength * normal_distribution[i]
        mutant_population[i] += np.array([x, y])
    return mutant_population


cube_constrains = _get_cube_constrains({
    "z1": [20, [1, 1]],
    "z2": [10, [-0.5, 1]],
    "z3": [5, [-1, -0.5]],
    "z4": [10, [1, -1]]
})

if __name__ == "__main__":
    print(cube_constrains)
    print(init_population(cube_constrains, 6))
    print(cost_function((0, 0)))
    print(cost_function((1, 0)))
    print(cost_function((10, 0)))
