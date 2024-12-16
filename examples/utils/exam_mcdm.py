#!/usr/bin/env python
# Created by "Thieu" at 09:19, 16/12/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from metamoo.utils.mcdm import weighted_sum, topsis, ahp, promethee, electre, vikor, moora, gra, goal_programming

pop_objs = np.array([
                    [1, 2, 3],
                    [2, 1.5, 4],
                    [2.5, 1, 3.5],
                    [3, 3, 5],
                    [3.5, 2.5, 2],
                    [2, 3, 4],
                    [4, 3, 5],
                    [1.5, 2.4, 3.4]])


# ---- Weighted Sum Model ----
# User needs to set up the weights (e.g, cost, time, accuracy)
weights = np.array([0.3, 0.2, 0.5])
is_benefit_objective = [False, False, True]  # f1, f2 are minimization, f3 is maximization

# Find the best solution by WSM
(best_idx, best_sol), (score, ) = weighted_sum(pop_objs, weights=weights, is_benefit_objective=is_benefit_objective)

print(f"\nBest Solution Selected by TOPSIS (f1, f2, f3):")
print(f"Best idx: {best_idx}, Best sol: {best_sol}")
print(f"The core list: {score}")


# ---- TOPSIS ----
# User needs to set up the weights (e.g, cost, time, accuracy)
weights = np.array([0.3, 0.2, 0.5])
is_benefit_objective = [False, False, True]  # f1, f2 are minimization, f3 is maximization

# Find the best solution by TOPSIS
(best_idx, best_sol), (score, )  = topsis(pop_objs, weights=weights, is_benefit_objective=is_benefit_objective)

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
(best_idx, best_sol), (score, ) = ahp(pop_objs, pairwise_matrix=pairwise_matrix, is_benefit_objective=is_benefit_objective)

print(f"\nBest Solution Selected by AHP (f1, f2, f3):")
print(f"Best idx: {best_idx}, Best sol: {best_sol}")
print(f"The core list: {score}")


# ---- PROMETHEE ----
# f1 (cost), f2 (time) and f3 (accuracy)
is_benefit_objective = [False, False, True]  # f1, f2 are minimization, f3 is maximization
weights = [0.2, 0.3, 0.5]  # Weights present importance for objectives

# Find the best solution by PROMETHEE
(best_idx, best_sol), (netflow, ) = promethee(pop_objs, weights, is_benefit_objective)

print(f"\nBest Solution Selected by PROMETHEE (f1, f2, f3):")
print(f"Best idx: {best_idx}, Best sol: {best_sol}")
print(f"The netflow list: {netflow}")


# ---- ELECTRE ----
# f1 (cost), f2 (time) and f3 (accuracy)
is_benefit_objective = [False, False, True]  # f1, f2 are minimization, f3 is maximization
weights = [0.2, 0.3, 0.5]  # Weights present importance for objectives

# Find the best solution by PROMETHEE
(best_idx, best_sol), (net_outflow, ) = electre(pop_objs, weights, is_benefit_objective)

print(f"\nBest Solution Selected by ELECTRE (f1, f2, f3):")
print(f"Best idx: {best_idx}, Best sol: {best_sol}")
print(f"The net outflow list: {net_outflow}")


# ---- VIKOR ----
# f1 (cost), f2 (time) and f3 (accuracy)
is_benefit_objective = [False, False, True]  # f1, f2 are minimization, f3 is maximization
weights = [0.2, 0.3, 0.5]  # Weights present importance for objectives

# Run VIKOR method
(best_idx, best_sol), (Q, rank_Q, rank_S, rank_R) = vikor(pop_objs, weights, is_benefit_objective, v=0.5)

# ---- Print Results ----
print(f"\nBest Solution Selected by VIKOR (f1, f2, f3):")
print(f"Best idx: {best_idx}, Best sol: {best_sol}")
print(f"The vikor Q index list: {Q}")

print("\nRanking of Solutions Based on VIKOR (Q):")
print(rank_Q)


# ---- MOORA ----
# f1 (cost), f2 (time) and f3 (accuracy)
is_benefit_objective = [False, False, True]  # f1, f2 are minimization, f3 is maximization
weights = [0.2, 0.3, 0.5]  # Weights present importance for objectives

# Run MOORA method
(best_idx, best_sol), (score, ranking) = moora(pop_objs, weights, is_benefit_objective)

# ---- Print Results ----
print(f"\nBest Solution Selected by MOORA (f1, f2, f3):")
print(f"Best idx: {best_idx}, Best sol: {best_sol}")
print(f"The score list: {score}")
print(f"The ranking list: {ranking}")


# ---- GRA ----
# f1 (cost), f2 (time) and f3 (accuracy)
is_benefit_objective = [False, False, True]  # f1, f2 are minimization, f3 is maximization
weights = [0.2, 0.3, 0.5]  # Weights present importance for objectives

# Run GRA method
(best_idx, best_sol), (score, ranking) = gra(pop_objs, weights, is_benefit_objective)

# ---- Print Results ----
print(f"\nBest Solution Selected by GRA (f1, f2, f3):")
print(f"Best idx: {best_idx}, Best sol: {best_sol}")
print(f"The score list: {score}")
print(f"The ranking list: {ranking}")


# ---- Goal Programming ----
# f1 (cost), f2 (time) and f3 (accuracy)
goals = [1.5, 4.0, 4.5]  # f1, f2 are minimization, f3 is maximization
weights = [1, 1, 1]  # Equal weights for deviations

# Run Goal Programming
(best_idx, best_sol), (min_deviation, )  = goal_programming(pop_objs, weights, goals)

# ---- Print Results ----
print(f"\nBest Solution Selected by Goal Programming (f1, f2, f3):")
print(f"Best idx: {best_idx}, Best sol: {best_sol}")
print(f"The total deviation from goals: {min_deviation}")
