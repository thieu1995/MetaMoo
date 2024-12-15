#!/usr/bin/env python
# Created by "Thieu" at 17:01, 15/12/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from metamoo import Optimizer, Population
from metamoo.utils.ref_point import generate_reference_points
from metamoo.core.selector import Nsga3Selector


class Nsga3(Optimizer):
    def __init__(self, epoch, pop_size, crossover=None, mutator=None, n_divisions=10,
                 seed=None, repairer=None, *args, **kwargs):
        super().__init__(seed, repairer, *args, **kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.crossover = crossover
        self.mutator = mutator
        self.n_divisions = n_divisions
        self.selector = Nsga3Selector(seed=seed)
        self.ref_points = None

    def initialization(self):
        self.pop = self.problem.generate_population(self.pop_size)
        self.ref_points = generate_reference_points(self.problem.n_objs, self.n_divisions)

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
        self.pop = Population(self.selector.do(pop_new, fronts_idx, self.pop_size, self.ref_points))
