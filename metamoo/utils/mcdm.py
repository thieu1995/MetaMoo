#!/usr/bin/env python
# Created by "Thieu" at 08:19, 16/12/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np


# ---- Normalize by norm before using any MCDM technique  ----
def get_normalize_by_norm(pop_objs):
    norm = np.linalg.norm(pop_objs, axis=0)
    return pop_objs / norm


# ----  Normalize by Min-Max values of the Objectives ----
def get_normalize_by_minmax(pop_objs):
    min_values = pop_objs.min(axis=0)
    max_values = pop_objs.max(axis=0)
    return (pop_objs - min_values) / (max_values - min_values)


def topsis(pop_objs, weights=None, is_benefit_objective=None):
    """
    Topsis method
    """
    # Normalize the matrix objectives
    normalized_matrix = get_normalize_by_norm(pop_objs)

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


def ahp(pop_objs, pairwise_matrix=None, is_benefit_objective=None):
    """
    Analytic Hierarchy Process method
    """
    # Normalize the pairwise matrix
    column_sums = np.sum(pairwise_matrix, axis=0)
    normalized_matrix = pairwise_matrix / column_sums

    # Calculate the priority vector (average of rows)
    weights = np.mean(normalized_matrix, axis=1)

    # ---- Adjust Objectives Based on Minimize/Maximize ----
    adjusted_matrix = np.copy(pop_objs)
    for idx in range(pop_objs.shape[1]):  # Iterate through each objective
        if is_benefit_objective[idx]:  # If maximize objective, negate values
            adjusted_matrix[:, idx] = -pop_objs[:, idx]

    # ---- Normalize the Objectives ----
    normalized_matrix = get_normalize_by_minmax(adjusted_matrix)

    # Compute scores using weighted sum
    scores = np.dot(normalized_matrix, weights)

    # Find the best solution
    best_solution_idx = np.argmax(scores)
    best_solution = pop_objs[best_solution_idx]

    return scores, best_solution, best_solution_idx

