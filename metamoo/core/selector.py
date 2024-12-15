#!/usr/bin/env python
# Created by "Thieu" at 8:57 AM, 16/10/2024 --------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from typing import List
from abc import ABC, abstractmethod
import numpy as np
from metamoo import Agent
from metamoo.utils.distance import calculate_crowding_distance


class Selector(ABC):
    @abstractmethod
    def do(self, agents:List[Agent], fronts=None, n_parents=None) -> List[Agent]:
        pass


class BinarySelector(Selector):
    def __init__(self, seed=None):
        self.generator = np.random.default_rng(seed)

    def do(self, agents:List[Agent], fronts=None, n_parents=None):
        selected_idx = self.generator.choice(len(agents), n_parents, replace=False)
        return [agents[idx] for idx in selected_idx]


class TournamentSelector(Selector):
    def __init__(self, k=2, seed=None):
        self.k = k
        self.generator = np.random.default_rng(seed)

    def do(self, agents:List[Agent], fronts=None, n_parents=None):
        obj_list = [agent.objectives for agent in agents]
        selected_parents = []
        while len(selected_parents) < n_parents:
            selected = self.generator.choice(len(agents), self.k, replace=False)
            selected_parents.append(min(selected, key=lambda idx: (fronts.index(next(f for f in fronts if idx in f)), obj_list[idx])))
        return [agents[idx] for idx in selected_parents]


class NsgaSelector(Selector):
    def __init__(self, seed=None):
        self.generator = np.random.default_rng(seed)

    def do(self, agents:List[Agent], fronts=None, n_parents=None):
        pop_selected = []
        for front in fronts:
            if len(pop_selected) + len(front) <= n_parents:
                pop_selected.extend(front)
            else:
                # Nếu thêm cả front sẽ vượt quá pop_size, chọn ngẫu nhiên từ front cuối cùng
                remaining_slots = n_parents - len(pop_selected)
                pop_selected.extend(self.generator.choice(front, remaining_slots, replace=False))
                break
        return [agents[idx] for idx in pop_selected]


class Nsga2Selector(Selector):
    def __init__(self, seed=None):
        self.generator = np.random.default_rng(seed)

    def do(self, agents:List[Agent], fronts=None, n_parents=None):
        crow_dist = calculate_crowding_distance(agents)
        pop_selected = []
        for front in fronts:
            if len(pop_selected) + len(front) <= n_parents:
                pop_selected.extend(front)
            else:
                sorted_front = sorted(front, key=lambda idx: crow_dist[idx], reverse=True)
                remaining_slots = n_parents - len(pop_selected)
                pop_selected.extend(sorted_front[:remaining_slots])
                break
        return [agents[idx] for idx in pop_selected]


class Nsga3Selector(Selector):
    def __init__(self, seed=None):
        self.generator = np.random.default_rng(seed)

    def do(self, agents:List[Agent], fronts=None, n_parents=None, ref_points=None):
        crow_dist = calculate_crowding_distance(agents)
        pop_selected = []
        for front in fronts:
            if len(pop_selected) + len(front) <= n_parents:
                pop_selected.extend(front)
            else:
                sorted_front = sorted(front, key=lambda idx: crow_dist[idx], reverse=True)
                remaining_slots = n_parents - len(pop_selected)
                pop_selected.extend(sorted_front[:remaining_slots])
                break
        return [agents[idx] for idx in pop_selected]


# # Base Layer for Selection, Crossover, and Mutation
# class Layer:
#     def __call__(self, *args, **kwargs):
#         raise NotImplementedError("Each layer must implement the __call__ method.")

# # Example Parent Selection Layer using tournament selection
# class NsgaSelection(Layer):
#     def __init__(self, n_agents=10):
#         self.n_agents = n_agents
#
#     def __call__(self, pareto_fronts, population, objectives):
#         selected_parents = []
#         for _ in range(len(population)):
#             # Perform tournament selection
#             candidates = np.random.choice(len(population), size=self.tournament_size, replace=False)
#             best_candidate = min(candidates, key=lambda x: objectives[x][0])
#             selected_parents.append(population[best_candidate])
#         return np.array(selected_parents)
#
#     # Chọn lọc cha mẹ từ các front dựa trên non-dominated sorting
#     def selection(population, fronts, pop_size):
#         selected_population = []
#         for front in fronts:
#             if len(selected_population) + len(front) <= pop_size:
#                 selected_population.extend(front)
#             else:
#                 # Nếu thêm cả front sẽ vượt quá pop_size, chọn ngẫu nhiên từ front cuối cùng
#                 remaining_slots = pop_size - len(selected_population)
#                 selected_population.extend(np.random.choice(front, remaining_slots, replace=False))
#                 break
#         return selected_population
#
#
#
# # Example Parent Selection Layer using tournament selection
# class TournamentSelection(Layer):
#     def __init__(self, tournament_size=2):
#         self.tournament_size = tournament_size
#
#     def __call__(self, pareto_fronts, population, objectives):
#         selected_parents = []
#         for _ in range(len(population)):
#             # Perform tournament selection
#             candidates = np.random.choice(len(population), size=self.tournament_size, replace=False)
#             best_candidate = min(candidates, key=lambda x: objectives[x][0])
#             selected_parents.append(population[best_candidate])
#         return np.array(selected_parents)
