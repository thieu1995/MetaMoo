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
from metamoo.utils.ref_point import associate_with_reference_points


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
        # Niching for Reference Point-based Selection
        ref_point_counts = np.zeros(ref_points.shape[0])
        pop_selected = []
        for front in fronts:
            if len(pop_selected) + len(front) <= n_parents:
                pop_selected.extend(front)
            else:
                pop_objs = np.array([agent.objectives for agent in agents])
                pop_objs = pop_objs[front]
                closest_points, _ = associate_with_reference_points(pop_objs, ref_points)
                unique_refs = np.unique(closest_points)
                while len(pop_selected) < n_parents:
                    ref_with_min_count = unique_refs[np.argmin(ref_point_counts[unique_refs])]
                    candidates = [i for i, ref in zip(front, closest_points) if ref == ref_with_min_count]
                    selected = candidates[self.generator.choice(len(candidates))]
                    pop_selected.append(selected)
                    ref_point_counts[ref_with_min_count] += 1
                break
        return [agents[idx] for idx in pop_selected]
