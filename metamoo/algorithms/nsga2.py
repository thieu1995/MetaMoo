#!/usr/bin/env python
# Created by "Thieu" at 10:24 AM, 10/10/2024 -------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from metamoo.core.prototype import Agent
from metamoo.utils.sorting import non_dominated_sorting


class NSGA2:
    def __init__(self, epoch, pop_size):
        self.epoch = epoch
        self.pop_size = pop_size
        self.problem, self.pop = None, None

    def solve(self, problem):
        self.problem = problem
        self.pop = self.problem.generate_population(self.pop_size)
        return self.evolve()
        
    def evolve(self):
        for generation in range(self.epoch):
            print(f"Generation {generation + 1}/{self.epoch}")
            
            # Non-dominated sorting
            fronts = non_dominated_sorting(self.pop)
            next_population = []
            
            # Generate the next population based on non-dominated sorting and crowding distance
            for front in fronts:
                if len(next_population) + len(front) > self.pop_size:
                    break
                next_population.extend(front)
            
            # Replace population with new generation
            self.pop = next_population

        return self.pop
