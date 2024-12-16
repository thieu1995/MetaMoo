#!/usr/bin/env python
# Created by "Thieu" at 08:19, 16/12/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np


# ---- Normalize before using any MCDM technique  ----
def get_normalize(pop_objs):
    norm = np.linalg.norm(pop_objs, axis=0)
    return pop_objs / norm


def topsis(pop_objs, weights, is_benefit_objective):
    # Normalize the matrix objectives
    normalized_matrix = get_normalize(pop_objs)

    # Apply the weights
    weighted_matrix = normalized_matrix * weights

    # Find the ideal solution max and min
    ideal_best = np.zeros(weighted_matrix.shape[1])
    ideal_worst = np.zeros(weighted_matrix.shape[1])
    for idx in range(weighted_matrix.shape[1]):
        if is_benefit_objective[idx]:
            ideal_best[idx] = np.max(weighted_matrix[:, idx])
            ideal_worst[idx] = np.min(weighted_matrix[:, idx])
        else:
            ideal_best[idx] = np.min(weighted_matrix[:, idx])
            ideal_worst[idx] = np.max(weighted_matrix[:, idx])

    # Calculate Euclidean
    dist_best = np.linalg.norm(weighted_matrix - ideal_best, axis=1)
    dist_worst = np.linalg.norm(weighted_matrix - ideal_worst, axis=1)

    # Find the TOPSIS score
    topsis_score = dist_worst / (dist_best + dist_worst)

    # Find the best solution by TOPSIS
    best_solution_idx = np.argmax(topsis_score)
    best_solution = pop_objs[best_solution_idx]

    return topsis_score, best_solution, best_solution_idx


