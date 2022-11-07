import json
from typing import Sequence, Tuple
from math import e as e_constant
from random import uniform, choice, randint

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
    # TODO: implement
    return population


def find_best_individual(
    population: Sequence, population_grades: Sequence
) -> Tuple[float, float]:
    min_index = 0
    min_grade = population_grades[min_index]
    for index in range(1, len(population_grades)):
        if population_grades[index] <= population_grades:
            min_index = index
            min_grade = population_grades[index]
    return population[min_index], min_grade


def one_point_crossover(population: Sequence, crossover_probability: float) -> Sequence:
    new_population = []
    for i in range(len(population) // 2):
        random_number = uniform(0, 1)
        if random_number < crossover_probability:
            new_population.append(population[i])
            new_population.append(population[i+1])
            continue
        first_parent = population[i]
        second_parent = population[i+1]
        intersection = randint(0, len(population))
        first_child = first_parent[:intersection] + second_parent[intersection:]
        second_child = first_parent[intersection:] + second_parent[:intersection]
        new_population.append(first_child)
        new_population.append(second_child)
    return new_population


def gauss_mutation(population: Sequence, mutation_strength: float) -> Sequence:
    # TODO : implement
    return population


cube_constrains = _get_cube_constrains(resources)

if __name__ == "__main__":
    print(cube_constrains)
    print(init_population(cube_constrains, 6))
    print(cost_function((0, 0)))
    print(cost_function((1, 0)))
    print(cost_function((10, 0)))
