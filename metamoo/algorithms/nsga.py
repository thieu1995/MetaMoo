#!/usr/bin/env python
# Created by "Thieu" at 9:13 AM, 16/10/2024 --------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from metamoo import Optimizer, Population


class NSGA(Optimizer):
    def __init__(self, epoch, pop_size, selector=None, crossover=None, mutator=None,
                 repairer=None, seed=None, *args, **kwargs):
        super().__init__(seed, repairer, *args, **kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.selector = selector
        self.crossover = crossover
        self.mutator = mutator

    def evolve(self, epoch):

        # Perform non-dominated sorting
        fronts_idx, _ = self.non_dominated_sorting(self.pop.agents)

        # Select parents for crossover
        pop_next = self.selector.do(self.pop.agents, fronts_idx, self.pop_size)      # List of agents

        pop_new = []
        for idx in range(0, self.pop_size):
            # Perform crossover to generate offspring
            parents = self.generator.choice(pop_next, 2, replace=False)
            agent = self.crossover.do(parents)

            # Perform mutation on offspring
            agent = self.mutator.do(agent)

            # Replace old population with offspring (you can add elitism or other strategies here)
            agent = self.repair_agent(agent)
            agent = self.evaluate_agent(agent)
            pop_new.append(agent)

        # Update population
        self.pop = Population(pop_new)
