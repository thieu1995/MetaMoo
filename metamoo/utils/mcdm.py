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


def get_concordance_matrix(matrix):
    """
    Compute the concordance matrix.
    """
    n_alternatives = matrix.shape[0]
    C = np.zeros((n_alternatives, n_alternatives))
    for idx in range(n_alternatives):
        for jdx in range(n_alternatives):
            if idx != jdx:
                C[idx, jdx] = np.sum(matrix[idx] >= matrix[jdx])
    return C


def get_discordance_matrix(matrix):
    """
    Compute the discordance matrix.
    """
    n_alternatives = matrix.shape[0]
    D = np.zeros((n_alternatives, n_alternatives))
    for idx in range(n_alternatives):
        for jdx in range(n_alternatives):
            if idx != jdx:
                D[idx, jdx] = np.max((matrix[jdx] - matrix[idx]) / (matrix.max(axis=0) - matrix.min(axis=0)))
    return D


def weighted_sum(pop_objs, weights=None, is_benefit_objective=None):
    """
    Weighted Sum Model (WSM) for multi-objective optimization.
    :param F: Pareto front solutions (n_solutions x n_objectives).
    :param weights: Weights for each objective.
    :param is_benefit_objective: List indicating if the objective is benefit-type (True) or cost-type (False).
    :return: Best solution and its aggregated score.
    """
    F_normalized = np.zeros_like(pop_objs)
    n_objectives = pop_objs.shape[1]

    # Normalize objectives (min-max normalization)
    for idx in range(n_objectives):
        f_min = pop_objs[:, idx].min()
        f_max = pop_objs[:, idx].max()

        if is_benefit_objective[idx]:  # Benefit-type objective
            F_normalized[:, idx] = (pop_objs[:, idx] - f_min) / (f_max - f_min)
        else:  # Cost-type objective
            F_normalized[:, idx] = (f_max - pop_objs[:, idx]) / (f_max - f_min)

    # Weighted sum calculation
    aggregated_scores = np.dot(F_normalized, weights)

    # Select the best solution (smallest aggregated score)
    best_solution_idx = np.argmin(aggregated_scores)
    best_solution = pop_objs[best_solution_idx]

    return (best_solution_idx, best_solution), (aggregated_scores, )


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

    return (best_solution_idx, best_solution), (topsis_score, )


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

    return (best_solution_idx, best_solution), (scores, )


def promethee(pop_objs, weights=None, is_benefit_objective=None):
    """
    Full PROMETHEE implementation: positive, negative, and net flow.
    """

    def preference_function(a, b, criterion_type="minimize"):
        """
        Define the preference function for pairwise comparison.
        """
        diff = b - a if criterion_type == "minimize" else a - b
        return max(0, diff)

    def calculate_preference_matrix(matrix, is_benefit_objective):
        """
        Calculate the preference matrix for all alternatives and objectives.
        """
        n_alternatives, n_objectives = matrix.shape
        preference_matrices = []
        # Loop through each objective
        for j in range(n_objectives):
            pref_matrix = np.zeros((n_alternatives, n_alternatives))
            for i in range(n_alternatives):
                for k in range(n_alternatives):
                    pref_matrix[i, k] = preference_function(
                        matrix[i, j], matrix[k, j],
                        criterion_type="maximize" if is_benefit_objective[j] else "minimize"
                    )
            preference_matrices.append(pref_matrix)
        return preference_matrices

    # Step 0: Adjust objectives for benefit/cost types
    adjusted_matrix = pop_objs.copy()
    for idx in range(pop_objs.shape[1]):
        if is_benefit_objective[idx]:
            adjusted_matrix[:, idx] = -pop_objs[:, idx]  # Negate to convert maximize to minimize

    # Step 1: Calculate preference matrices
    preference_matrices = calculate_preference_matrix(adjusted_matrix, is_benefit_objective)
    n_alternatives = adjusted_matrix.shape[0]

    # Step 2: Aggregate preference indices
    aggregated_pref = np.zeros((n_alternatives, n_alternatives))
    for jdx in range(len(weights)):
        aggregated_pref += weights[jdx] * preference_matrices[jdx]

    # Step 3: Compute positive and negative flows
    positive_flow = np.sum(aggregated_pref, axis=1) / (n_alternatives - 1)
    negative_flow = np.sum(aggregated_pref, axis=0) / (n_alternatives - 1)

    # Step 4: Calculate netflow
    net_flow = positive_flow - negative_flow

    # Step 5: Rank Solutions to fine the best
    best_solution_idx = np.argmax(net_flow)
    best_solution = pop_objs[best_solution_idx]

    return (best_solution_idx, best_solution), (net_flow, )


def electre(pop_objs, weights=None, is_benefit_objective=None, concordance_threshold=0.6, discordance_threshold=0.4):
    """
    ELECTRE method implementation.
    """
    # Step 0: Adjust objectives for benefit/cost types
    adjusted_matrix = pop_objs.copy()
    for idx in range(pop_objs.shape[1]):
        if is_benefit_objective[idx]:
            adjusted_matrix[:, idx] = -pop_objs[:, idx]  # Negate to convert maximize to minimize

    # Step 1: Normalize and weight the matrix
    norm_matrix = get_normalize_by_norm(adjusted_matrix)
    weighted_mat = norm_matrix * weights

    # Step 2: Calculate concordance and discordance matrices
    C = get_concordance_matrix(weighted_mat)
    D = get_discordance_matrix(norm_matrix)

    # Step 3: Apply thresholds
    n_alternatives = adjusted_matrix.shape[0]
    S = np.zeros((n_alternatives, n_alternatives))
    for idx in range(n_alternatives):
        for jdx in range(n_alternatives):
            if idx != jdx and C[idx, jdx] >= concordance_threshold and D[idx, jdx] <= discordance_threshold:
                S[idx, jdx] = 1  # Alternative i dominates j

    # Step 4: Rank solutions
    net_outflow = np.sum(S, axis=1)  # Count of alternatives dominated by each solution

    # Step 5: Rank Solutions to fine the best
    best_solution_idx = np.argmax(net_outflow)
    best_solution = pop_objs[best_solution_idx]

    return (best_solution_idx, best_solution), (net_outflow, )


def vikor(pop_objs, weights=None, is_benefit_objective=None, v=0.5):
    """
    VIKOR method implementation.
    :param matrix: Decision matrix (alternatives x objectives).
    :param weights: List of weights for each objective.
    :param is_benefit_objective: List indicating which objectives are benefit (True) or cost (False).
    :param v: VIKOR balancing parameter (0 ≤ v ≤ 1).
    :return: Best alternative index and rankings.
    """
    n_alternatives, n_criteria = pop_objs.shape

    # Step 1: Adjust objectives based on benefit/cost types
    adjusted_matrix = pop_objs.copy()
    for idx in range(n_criteria):
        if is_benefit_objective[idx]:
            adjusted_matrix[:, idx] = -pop_objs[:, idx]  # Convert maximize to minimize

    # Step 2: Identify best and worst values for each criterion
    f_best = np.min(adjusted_matrix, axis=0)
    f_worst = np.max(adjusted_matrix, axis=0)

    # Step 3: Calculate S_i (Utility) and R_i (Regret) for each alternative
    S = np.zeros(n_alternatives)
    R = np.zeros(n_alternatives)

    for idx in range(n_alternatives):
        S[idx] = np.sum(weights * (adjusted_matrix[idx] - f_best) / (f_worst - f_best))
        R[idx] = np.max(weights * (adjusted_matrix[idx] - f_best) / (f_worst - f_best))

    # Step 4: Calculate Q_i (VIKOR index)
    S_best, S_worst = np.min(S), np.max(S)
    R_best, R_worst = np.min(R), np.max(R)

    Q = v * (S - S_best) / (S_worst - S_best) + (1 - v) * (R - R_best) / (R_worst - R_best)

    # Step 5: Rank alternatives based on Q, S, and R
    rank_Q = np.argsort(Q)
    rank_S = np.argsort(S)
    rank_R = np.argsort(R)

    best_solution_idx = rank_Q[0]
    best_solution = pop_objs[best_solution_idx]

    return (best_solution_idx, best_solution), (Q, rank_Q, rank_S, rank_R)


def moora(pop_objs, weights=None, is_benefit_objective=None):
    """
    MOORA method implementation.
    :param matrix: Decision matrix (alternatives x objectives).
    :param weights: List of weights for each objective.
    :param is_benefit_objective: List indicating which objectives are benefit (True) or cost (False).
    :return: MOORA scores and ranked alternatives.
    """
    n_alternatives, n_criteria = pop_objs.shape

    # Step 1: Normalize the decision matrix
    normalized_matrix = get_normalize_by_norm(pop_objs)

    # Step 2: Apply weights to the normalized matrix
    weighted_matrix = normalized_matrix * weights

    # Step 3: Calculate MOORA ratios
    scores = np.zeros(n_alternatives)
    for idx in range(n_criteria):
        if is_benefit_objective[idx]:
            scores += weighted_matrix[:, idx]  # Benefit criteria (maximize)
        else:
            scores -= weighted_matrix[:, idx]  # Cost criteria (minimize)

    # Step 4: Rank alternatives based on scores
    ranking = np.argsort(-scores)  # Sort in descending order of scores

    best_solution_idx = ranking[0]
    best_solution = pop_objs[best_solution_idx]

    return (best_solution_idx, best_solution), (scores, ranking)


def gra(pop_objs, weights=None, is_benefit_objective=None):
    """
    Gray Relational Analysis (GRA) method implementation.
    :param matrix: Decision matrix (alternatives x objectives).
    :param weights: List of weights for each objective.
    :param is_benefit_objective: List indicating which objectives are benefit (True) or cost (False).
    :return: Gray relational grades and ranked alternatives.
    """
    n_alternatives, n_criteria = pop_objs.shape

    # Step 1: Normalize the decision matrix
    normalized_matrix = np.zeros_like(pop_objs)
    for j in range(n_criteria):
        if is_benefit_objective[j]:
            normalized_matrix[:, j] = (pop_objs[:, j] - np.min(pop_objs[:, j])) / (np.max(pop_objs[:, j]) - np.min(pop_objs[:, j]))
        else:
            normalized_matrix[:, j] = (np.max(pop_objs[:, j]) - pop_objs[:, j]) / (np.max(pop_objs[:, j]) - np.min(pop_objs[:, j]))

    # Step 2: Determine the ideal solution (all 1s after normalization)
    ideal_solution = np.ones(n_criteria)

    # Step 3: Calculate Gray Relational Coefficients
    epsilon = 1e-6  # Small constant to avoid division by zero
    diff_matrix = np.abs(normalized_matrix - ideal_solution)
    min_diff = np.min(diff_matrix)
    max_diff = np.max(diff_matrix)

    gray_relational_coefficients = (min_diff + epsilon) / (diff_matrix + max_diff + epsilon)

    # Step 4: Calculate Gray Relational Grades
    gray_relational_grades = np.sum(gray_relational_coefficients * weights, axis=1)

    # Step 5: Rank alternatives
    ranking = np.argsort(-gray_relational_grades)  # Sort in descending order of grades

    best_solution_idx = ranking[0]
    best_solution = pop_objs[best_solution_idx]

    return (best_solution_idx, best_solution), (gray_relational_grades, ranking)


def goal_programming(pop_objs, weights=None, goals=None):
    """
    Goal Programming implementation to minimize deviations from predefined goals.
    :param F: Decision matrix (Pareto front solutions).
    :param goals: Target goals for each objective.
    :param weights: Weights for each objective's deviations.
    :return: Best solution and deviations.
    """
    from scipy.optimize import linprog

    n_solutions, n_objectives = pop_objs.shape

    # Objective function: Minimize weighted sum of deviations (d+ and d-)
    c = []
    A_eq = []
    b_eq = []
    bounds = []

    # Add deviation variables for each objective (positive and negative)
    for idx in range(n_objectives):
        c.extend([weights[idx], weights[idx]])  # Weights for d- and d+
        bounds.extend([(0, None), (0, None)])  # Deviations are non-negative

    # Add equality constraints for each objective
    for idx in range(n_objectives):
        constraint = np.zeros(n_objectives * 2)
        constraint[2 * idx] = 1         # d-
        constraint[2 * idx + 1] = -1    # d+
        A_eq.append(constraint)
        b_eq.append(goals[idx])

    # Add decision variables (actual solutions) and their deviations
    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)

    # Solve linear programming for each solution in Pareto front
    best_solution = None
    best_index = None
    min_deviation = float("inf")

    for idx, solution in enumerate(pop_objs):
        # Modify RHS to include the actual objective values
        b_eq_modified = b_eq - solution
        res = linprog(c, A_eq=A_eq, b_eq=b_eq_modified, bounds=bounds, method='highs')

        if res.success:
            total_deviation = res.fun
            if total_deviation < min_deviation:
                min_deviation = total_deviation
                best_solution = solution
                best_index = idx

    return (best_index, best_solution), (min_deviation, )

