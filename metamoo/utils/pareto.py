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
