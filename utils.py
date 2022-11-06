import json
from typing import Sequence
from math import e as e_constant
from random import uniform

with open("resources.json") as f:
    resources = json.load(f)


def _manhattandistance(x: Sequence[float], y: Sequence[float]) -> float:
    return abs(x[0] - y[0]) + abs(x[1] - y[1])


def cost_function(resources: dict, localisation: Sequence[float]) -> float:
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


cube_constains = _get_cube_constrains(resources)

if __name__ == "__main__":
    print(cube_constains)
    print(init_population(cube_constains, 6))
    print(cost_function(resources, (0, 0)))
    print(cost_function(resources, (1, 0)))
    print(cost_function(resources, (10, 0)))