#!/usr/bin/env python
# Created by "Thieu" at 11:01 AM, 16/10/2024 -------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from abc import ABC, abstractmethod
import numpy as np
from metamoo import Agent


class Mutator(ABC):

    def __init__(self, kind="single", mutation_rate=0.05, seed=None, **kwargs):
        self.kind = kind
        self.mutation_rate = mutation_rate
        self.seed = seed
        self.generator = np.random.default_rng(seed)

    @abstractmethod
    def do(self, agent: Agent) -> Agent:
        pass


class SwapMutator(Mutator):
    def __init__(self, kind="single", mutation_rate=0.1, seed=None, **kwargs):
        super().__init__(kind, mutation_rate, seed, **kwargs)

    def do(self, agent: Agent) -> Agent:
        if self.kind == "single":
            if self.generator.random() < self.mutation_rate:
                j1, j2 = self.generator.choice(range(0, len(agent.solution)), 2, replace=False)
                agent.solution[j1], agent.solution[j2] = agent.solution[j2], agent.solution[j1]
            return agent
        else:   # multi-point
            for idx in range(len(agent.solution)):
                if self.generator.random() < self.mutation_rate:
                    idx_swap = self.generator.choice(list(set(range(0, len(agent.solution))) - {idx}))
                    agent.solution[idx], agent.solution[idx_swap] = agent.solution[idx_swap], agent.solution[idx]
            return agent


class UniformFlipMutator(Mutator):
    def __init__(self, kind="single", mutation_rate=0.1, seed=None, lb=None, ub=None, **kwargs):
        super().__init__(kind, mutation_rate, seed, **kwargs)
        self.lb = lb
        self.ub = ub

    def do(self, agent: Agent) -> Agent:
        if self.kind == "single":
            if self.generator.random() < self.mutation_rate:
                idx = self.generator.integers(0, len(agent.solution))
                agent.solution[idx] = self.generator.uniform(self.lb[idx], self.ub[idx])
            return agent
        else:   # multi-point
            mutation_child = self.generator.uniform(self.lb, self.ub)
            flag_child = self.generator.uniform(0, 1, len(agent.solution)) < self.mutation_rate
            vec = np.where(flag_child, mutation_child, agent.solution)
            agent.solution = vec
            return agent


class GaussianFlipMutator(Mutator):
    def __init__(self, kind="single", mutation_rate=0.1, seed=None, loc=None, scale=None, **kwargs):
        super().__init__(kind, mutation_rate, seed, **kwargs)
        self.loc = 0 if loc is None else loc
        self.scale = 1 if scale is None else scale

    def do(self, agent: Agent) -> Agent:
        self.loc = self.loc * np.ones(len(agent.solution))
        self.scale = self.scale * np.ones(len(agent.solution))
        if self.kind == "single":
            if self.generator.random() < self.mutation_rate:
                idx = self.generator.integers(0, len(agent.solution))
                agent.solution[idx] = self.generator.normal(self.loc[idx], self.scale[idx])
            return agent
        else:   # multi-point
            mutation_child = self.generator.normal(self.loc, self.scale)
            flag_child = self.generator.uniform(0, 1, len(agent.solution)) < self.mutation_rate
            vec = np.where(flag_child, mutation_child, agent.solution)
            agent.solution = vec
            return agent


class InversionMutator(Mutator):
    def __init__(self, kind="single", mutation_rate=0.1, seed=None, **kwargs):
        super().__init__(kind, mutation_rate, seed, **kwargs)

    def do(self, agent: Agent) -> Agent:
        if self.generator.random() < self.mutation_rate:
            cut1, cut2 = self.generator.choice(range(0, len(agent.solution)), 2, replace=False)
            temp = agent.solution[cut1:cut2]
            agent.solution[cut1:cut2] = temp[::-1]
        return agent


class ScrambleMutator(Mutator):
    def __init__(self, kind="single", mutation_rate=0.1, seed=None, **kwargs):
        super().__init__(kind, mutation_rate, seed, **kwargs)

    def do(self, agent: Agent) -> Agent:
        if self.generator.random() < self.mutation_rate:
            cut1, cut2 = self.generator.choice(range(0, len(agent.solution)), 2, replace=False)
            temp = agent.solution[cut1:cut2]
            self.generator.shuffle(temp)
            agent.solution[cut1:cut2] = temp
        return agent
