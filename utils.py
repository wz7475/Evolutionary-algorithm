import copy
import json
from typing import Sequence, Tuple
from math import e as e_constant
from random import uniform, choices, randint
import numpy as np

with open("resources.json") as f:
    resources = json.load(f)


def _manhattandistance(x: Sequence[float], y: Sequence[float]) -> float:
    return abs(x[0] - y[0]) + abs(x[1] - y[1])


def cost_function(localisation: Sequence[float]) -> float:
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


def roulette_reproduction(population: Sequence, population_grades: Sequence):
    distribution = [1 - grade / sum(population_grades) for grade in population_grades]
    return choices(population=population, weights=distribution, k=len(population))


def find_best_individual(
    population: Sequence, population_grades: Sequence
) -> Tuple[float, float]:
    min_index = 0
    min_grade = population_grades[min_index]
    for index in range(1, len(population_grades)):
        if population_grades[index] <= population_grades[index]:
            min_index = index
            min_grade = population_grades[index]
    return population[min_index], min_grade


def one_point_crossover(population: Sequence, crossover_probability: float) -> Sequence:
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


cube_constrains = _get_cube_constrains(resources)

if __name__ == "__main__":
    print(cube_constrains)
    print(init_population(cube_constrains, 6))
    print(cost_function((0, 0)))
    print(cost_function((1, 0)))
    print(cost_function((10, 0)))
