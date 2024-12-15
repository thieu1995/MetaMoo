#!/usr/bin/env python
# Created by "Thieu" at 9:13 AM, 16/10/2024 --------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from metamoo import Optimizer, Population
from metamoo.core.selector import NsgaSelector


class Nsga(Optimizer):
    def __init__(self, epoch, pop_size, crossover=None, mutator=None,
                 repairer=None, seed=None, *args, **kwargs):
        super().__init__(seed, repairer, *args, **kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.crossover = crossover
        self.mutator = mutator
        self.selector = NsgaSelector(seed=seed)

    def evolve(self, epoch):

        pop_new = []
        for idx in range(0, self.pop_size):
            # Perform crossover to generate offspring
            parents = self.generator.choice(self.pop.agents, 2, replace=False)
            agent = self.crossover.do(parents)

            # Perform mutation on offspring
            agent = self.mutator.do(agent)

            # Replace old population with offspring (you can add elitism or other strategies here)
            agent = self.repair_agent(agent)
            agent = self.evaluate_agent(agent)
            pop_new.append(agent)

        # Create a pool of agents
        pop_new.extend(self.pop.agents)

        # Perform non-dominated sorting
        fronts_idx, _ = self.non_dominated_sorting(pop_new)

        # Select parents for next generation
        self.pop = Population(self.selector.do(pop_new, fronts_idx, self.pop_size))
