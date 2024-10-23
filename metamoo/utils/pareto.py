#!/usr/bin/env python
# Created by "Thieu" at 10:24 AM, 10/10/2024 -------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from typing import List
import numpy as np
from metamoo.core.prototype import Agent


def dominates(agent1: Agent, agent2: Agent):
    """
    Check if the objectives of obj1 dominate the objectives of obj2 using NumPy.

    obj1 and obj2 are arrays (or lists) of objective values.
    """
    obj1 = np.array(agent1.objectives)
    obj2 = np.array(agent2.objectives)
    # Check if obj1 is better or equal in all objectives
    better_or_equal = np.all(obj1 <= obj2)
    # Check if obj1 is strictly better in at least one objective
    strictly_better = np.any(obj1 < obj2)
    return better_or_equal and strictly_better


def classify_agents(agents: List[Agent]):
    """Classify solutions into non-dominated and dominated sets"""
    non_dominated = []
    dominated = []
    for agent1 in agents:
        is_dominated = False
        for agent2 in agents:
            if agent1 != agent2 and dominates(agent2, agent1):
                is_dominated = True
                break
        if is_dominated:
            dominated.append(agent1.solution)
        else:
            non_dominated.append(agent1.solution)
    return non_dominated, dominated


def non_dominated_sorting(agents: List[Agent]):
    """
    :param agents:
    :return: List indexes of pareto fronts and list agents
    """
    obj_list = np.array([agent.objectives for agent in agents]).T       # Shape = (n_objs, n_agents)
    pop_size = obj_list.shape[1]  # Number of solutions in the population
    fronts = [[]]  # List of fronts, each front is a list of solution indices
    domination_count = np.zeros(pop_size, dtype=int)  # Number of solutions that dominate each solution
    dominated_solutions = [[] for _ in range(pop_size)]  # List of solutions that are dominated by each solution

    # Determine the domination relationships
    for i in range(pop_size):
        for j in range(i + 1, pop_size):
            if (obj_list[:, i] <= obj_list[:, j]).all() and (obj_list[:, i] < obj_list[:, j]).any():
                # Solution i dominates solution j
                dominated_solutions[i].append(j)
                domination_count[j] += 1
            elif (obj_list[:, j] <= obj_list[:, i]).all() and (obj_list[:, j] < obj_list[:, i]).any():
                #  Solution j dominates solution i
                dominated_solutions[j].append(i)
                domination_count[i] += 1

        # If no solutions dominate solution i, add it to the first front
        if domination_count[i] == 0:
            fronts[0].append(i)

    front_idx = 0  # Current front index
    # Generate the subsequent fronts
    while len(fronts[front_idx]) > 0:
        next_front = []
        for i in fronts[front_idx]:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                # If solution j is no longer dominated by any other solutions, add it to the next front
                if domination_count[j] == 0:
                    next_front.append(j)
        front_idx += 1
        fronts.append(next_front)

    fronts_new = []
    for front in fronts[:-1]:
        fronts_new.append([agents[i] for i in front])
    # # Return all fronts except the last empty one
    return fronts[:-1], fronts_new      # list indexes of pareto fronts and list agents
