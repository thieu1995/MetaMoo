#!/usr/bin/env python
# Created by "Thieu" at 10:24 AM, 10/10/2024 -------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

# def non_dominated_sorting(population):
#     # Simple non-dominated sorting logic
#     fronts = []
#     front = []
#     for Agent in population:
#         is_dominated = False
#         for other_ind in population:
#             if dominates(other_ind, Agent):
#                 is_dominated = True
#                 break
#         if not is_dominated:
#             front.append(Agent)
#     fronts.append(front)
#     return fronts
#
# def dominates(ind1, ind2):
#     return all(x <= y for x, y in zip(ind1.objectives, ind2.objectives)) and any(x < y for x, y in zip(ind1.objectives, ind2.objectives))

import numpy as np

def non_dominated_sorting(population):
    fronts = []
    front = []
    for agent in population:
        is_dominated = False
        for other_ind in population:
            if dominates(other_ind, agent):
                is_dominated = True
                break
        if not is_dominated:
            front.append(agent)
    fronts.append(front)
    return fronts

def dominates(ind1, ind2):
    return np.all(ind1.objectives <= ind2.objectives) and np.any(ind1.objectives < ind2.objectives)
