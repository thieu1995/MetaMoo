#!/usr/bin/env python
# Created by "Thieu" at 10:01 AM, 16/10/2024 -------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from typing import List
from abc import ABC, abstractmethod
import numpy as np
from metamoo import Agent


class Crossover(ABC):
    def __init__(self, crossover_rate=0.8, seed=None, **kwargs):
        self.crossover_rate = crossover_rate
        self.generator = np.random.default_rng(seed)

    @abstractmethod
    def do(self, agents: List[Agent]) -> Agent:
        pass


class ArithmeticCrossover(Crossover):
    def __init__(self, crossover_rate=0.8, seed=None, **kwargs):
        super().__init__(crossover_rate, seed, **kwargs)

    def do(self, parents: List[Agent]) -> Agent:
        if self.generator.random() > self.crossover_rate:
            return parents[0] if self.generator.random() < 0.5 else parents[1]

        rr = self.generator.uniform()
        vec1 = np.multiply(rr, parents[0].solution) + np.multiply((1 - rr), parents[1].solution)
        vec2 = np.multiply(rr, parents[1].solution) + np.multiply((1 - rr), parents[0].solution)
        agent1 = Agent(solution=vec1)
        agent2 = Agent(solution=vec2)
        return agent1 if self.generator.random() < 0.5 else agent2


class UniformCrossover(Crossover):
    def __init__(self, crossover_rate=0.8, seed=None, **kwargs):
        super().__init__(crossover_rate, seed, **kwargs)

    def do(self, parents: List[Agent]) -> Agent:
        if self.generator.random() > self.crossover_rate:
            return parents[0] if self.generator.random() < 0.5 else parents[1]

        mask = self.generator.integers(0, 2, size=len(parents[0].solution), dtype=bool)
        vec1 = np.where(mask, parents[0].solution, parents[1].solution)
        vec2 = np.where(mask, parents[1].solution, parents[0].solution)
        agent1 = Agent(solution=vec1)
        agent2 = Agent(solution=vec2)
        return agent1 if self.generator.random() < 0.5 else agent2


class OnePointCrossover(Crossover):
    def __init__(self, crossover_rate=0.8, seed=None, **kwargs):
        super().__init__(crossover_rate, seed, **kwargs)

    def do(self, parents: List[Agent]) -> Agent:
        if self.generator.random() > self.crossover_rate:
            return parents[0] if self.generator.random() < 0.5 else parents[1]

        point = self.generator.choice(range(0, len(parents[0].solution)))
        vec1 = np.concatenate([parents[0].solution[:point], parents[1].solution[point:]])
        vec2 = np.concatenate([parents[1].solution[:point], parents[0].solution[point:]])
        agent1 = Agent(solution=vec1)
        agent2 = Agent(solution=vec2)
        return agent1 if self.generator.random() < 0.5 else agent2


class MultiPointsCrossover(Crossover):
    def __init__(self, crossover_rate=0.8, seed=None, **kwargs):
        super().__init__(crossover_rate, seed, **kwargs)

    def do(self, parents: List[Agent]) -> Agent:
        if self.generator.random() > self.crossover_rate:
            return parents[0] if self.generator.random() < 0.5 else parents[1]

        idxs = self.generator.choice(range(0, len(parents[0].solution)), 2, replace=False)
        cut1, cut2 = np.min(idxs), np.max(idxs)
        vec1 = np.concatenate([parents[0].solution[:cut1], parents[1].solution[cut1:cut2], parents[0][cut2:]])
        vec2 = np.concatenate([parents[1].solution[:cut1], parents[0].solution[cut1:cut2], parents[1][cut2:]])
        agent1 = Agent(solution=vec1)
        agent2 = Agent(solution=vec2)
        return agent1 if self.generator.random() < 0.5 else agent2
