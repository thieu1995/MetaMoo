#!/usr/bin/env python
# Created by "Thieu" at 15:09, 13/12/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np


# Crowding distance calculation
def calculate_crowding_distance(list_agents=None):
    pop_objs = np.array([agent.objectives for agent in list_agents])
    n_agents, n_objs = pop_objs.shape       # (n_agents, n_objs)
    dist = np.zeros(n_agents)

    for m in range(n_objs):
        idx_sorted = np.argsort(pop_objs[:, m])
        dist[idx_sorted[0]] = np.inf
        dist[idx_sorted[-1]] = np.inf
        space = pop_objs[idx_sorted[-1], m] - pop_objs[idx_sorted[0], m]
        for idx in range(1, n_agents - 1):
            dist[idx_sorted[idx]] += (pop_objs[idx_sorted[idx + 1], m] - pop_objs[idx_sorted[idx - 1], m]) / space
    return dist
