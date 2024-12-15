#!/usr/bin/env python
# Created by "Thieu" at 01:26, 15/12/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from metamoo.core.prototype import Agent
from metamoo.utils.distance import calculate_crowding_distance


def calculate_crowding_distance2(population, objectives):

    num_individuals = len(population)
    num_objectives = objectives.shape[1]

    # Initialize crowding distance to 0
    crowding_distance = np.zeros(num_individuals)

    # Iterate through each objective
    for m in range(num_objectives):
        # Sort individuals based on the values of objective m
        sorted_indices = np.argsort(objectives[:, m])
        sorted_objectives = objectives[sorted_indices, m]

        # Assign large distance to boundary individuals
        crowding_distance[sorted_indices[0]] = np.inf
        crowding_distance[sorted_indices[-1]] = np.inf

        # Calculate crowding distance for the remaining individuals
        for i in range(1, num_individuals - 1):
            if sorted_objectives[-1] - sorted_objectives[0] == 0:  # Avoid division by zero
                norm = 1.0
            else:
                norm = sorted_objectives[-1] - sorted_objectives[0]

            crowding_distance[sorted_indices[i]] += (
                    (sorted_objectives[i + 1] - sorted_objectives[i - 1]) / norm
            )

    return crowding_distance


# Example usage
if __name__ == "__main__":
    # Assume 5 individuals with 2 objectives
    objectives = np.array([[1, 2, 3],
                           [2, 1.5, 4],
                           [2.5, 1, 3.5],
                           [3, 3, 5],
                           [3.5, 2.5, 2],
                           [2, 3, 4],
                           [4, 3, 5],
                           [1.5, 2.4, 3.4]])
    population = [0, 1, 2, 3, 4, 5, 6, 7]

    distances = calculate_crowding_distance2(population, objectives)
    print("Crowding distances:", distances)

    list_agents = []
    for obj in objectives:
        list_agents.append(Agent(objectives=obj))
    print(calculate_crowding_distance(list_agents))
