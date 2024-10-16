#!/usr/bin/env python
# Created by "Thieu" at 2:41 PM, 16/10/2024 -------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from abc import ABC, abstractmethod
import numpy as np
from metamoo import Agent


class Repairer(ABC):

    @abstractmethod
    def do(self, agent: Agent) -> Agent:
        pass


class BoundRepair(Repairer):
    def __init__(self, lb=None, ub=None):
        self.lb = lb
        self.ub = ub

    def do(self, agent: Agent) -> Agent:
        sol = np.clip(agent.solution, self.lb, self.ub)
        agent.solution = sol
        return agent


class UniformRandomRepair(Repairer):
    def __init__(self, lb=None, ub=None, seed=None):
        self.lb = lb
        self.ub = ub
        self.generator = np.random.default_rng(seed)

    def do(self, agent: Agent) -> Agent:
        sol = agent.solution
        mask = (sol < self.lb) | (sol > self.ub)
        sol[mask] = self.generator.uniform(self.lb[mask], self.ub[mask])
        agent.solution = sol
        return agent


class GaussianRandomRepair(Repairer):
    def __init__(self, loc=None, scale=None, lb=None, ub=None, seed=None):
        self.loc = loc
        self.scale = scale
        if loc is None:
            self.loc = np.zeros(len(lb))
        if scale is None:
            self.scale = np.ones(len(lb))
        self.lb = lb
        self.ub = ub
        self.generator = np.random.default_rng(seed)

    def do(self, agent: Agent) -> Agent:
        sol = agent.solution
        mask = (sol < self.lb) | (sol > self.ub)
        sol[mask] = self.generator.normal(self.loc[mask], self.ub[mask])
        sol = np.clip(sol, self.lb, self.ub)
        agent.solution = sol
        return agent


class CircularRepair(Repairer):
    def __init__(self, lb=None, ub=None):
        self.lb = lb
        self.ub = ub

    def do(self, agent: Agent) -> Agent:
        range_size = self.ub - self.lb
        repaired_solution = self.lb + (agent.solution - self.lb) % range_size
        agent.solution = repaired_solution
        return agent


# class ReverseRepair(Repairer):
#     def __init__(self, lb=None, ub=None):
#         self.lb = lb
#         self.ub = ub
#
#     def repair(self, offspring):
#         for i in range(len(offspring)):
#             for j in range(len(offspring[i])):
#                 if offspring[i][j] > self.ub[j]:
#                     offspring[i][j] = self.ub[j]
#                 if offspring[i][j] < self.lb[j]:
#                     offspring[i][j] = self.lb[j]
#                 return offspring
#
#
# class RoundRepair(Repairer):
#
#     def __init__(self, step=1.0):
#         self.step = step
#
#     def repair(self, offspring):
#         return np.round(offspring / self.step) * self.step
#
#
# class EvenOddRepair(Repairer):
#
#     def __init__(self, even=True):
#         self.even = even
#
#     def repair(self, offspring):
#         for i in range(len(offspring)):
#             for j in range(len(offspring[i])):
#                 if self.even and offspring[i][j] % 2 != 0:
#                     offspring[i][j] += 1
#                 elif not self.even and offspring[i][j] % 2 == 0:
#                     offspring[i][j] += 1
#         return offspring

