#!/usr/bin/env python
# Created by "Thieu" at 09:19, 16/12/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from metamoo.utils.mcdm import topsis, ahp, promethee

pop_objs = np.array([
                    [1, 2, 3],
                    [2, 1.5, 4],
                    [2.5, 1, 3.5],
                    [3, 3, 5],
                    [3.5, 2.5, 2],
                    [2, 3, 4],
                    [4, 3, 5],
                    [1.5, 2.4, 3.4]])

# ---- TOPSIS ----
# User needs to set up the weights (e.g, cost, time, accuracy)
weights = np.array([0.3, 0.2, 0.5])
is_benefit_objective = [False, False, True]  # f1, f2 are minimization, f3 is maximization

# Find the best solution by TOPSIS
score, best_sol, best_idx = topsis(pop_objs, weights=weights, is_benefit_objective=is_benefit_objective)

print(f"\nBest Solution Selected by TOPSIS (f1, f2, f3):")
print(f"Best idx: {best_idx}, Best sol: {best_sol}")
print(f"The core list: {score}")


# ---- AHP ----
# Example: Pairwise comparison matrix for 3 objectives
# f1 (cost), f2 (time) and f3 (accuracy): Suppose f3 and f2 is moderately more important
pairwise_matrix = np.array([
    [1, 1/2, 1/3],      # f1 compared to itself and f2, f3
    [2, 1,   1/2],      # f2 compared to f1, itself, and f3
    [3, 2,     1]       # f3 compared to f1, f2 and itself
])
is_benefit_objective = [False, False, True]  # f1, f2 are minimization, f3 is maximization

# Find the best solution by AHP
score, best_sol, best_idx = ahp(pop_objs, pairwise_matrix=pairwise_matrix, is_benefit_objective=is_benefit_objective)

print(f"\nBest Solution Selected by AHP (f1, f2, f3):")
print(f"Best idx: {best_idx}, Best sol: {best_sol}")
print(f"The core list: {score}")


# ---- PROMETHEE ----
# f1 (cost), f2 (time) and f3 (accuracy)
is_benefit_objective = [False, False, True]  # f1, f2 are minimization, f3 is maximization
weights = [0.2, 0.3, 0.5]  # Equal importance for both objectives

# Find the best solution by PROMETHEE
netflow, best_sol, best_idx = promethee(pop_objs, weights, is_benefit_objective)

print(f"\nBest Solution Selected by AHP (f1, f2, f3):")
print(f"Best idx: {best_idx}, Best sol: {best_sol}")
print(f"The netflow list: {netflow}")



