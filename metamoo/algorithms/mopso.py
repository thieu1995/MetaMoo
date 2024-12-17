#!/usr/bin/env python
# Created by "Thieu" at 15:27, 17/12/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from metamoo import Optimizer, Population, Agent


class MOPSO(Optimizer):
    def __init__(self, epoch, pop_size, inertia_weight=0.5, c1=1.5, c2=1.5,
                 seed=None, repairer=None, *args, **kwargs):
        super().__init__(seed, repairer, *args, **kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.inertia_weight = inertia_weight    # Fixed inertia
        self.c1 = c1
        self.c2 = c2    # Cognitive and social coefficients

    def post_initialization_hook(self):
        # Initialize velocities randomly
        self.vec_bound = (self.problem.ub - self.problem.lb) * 0.1
        self.velocities = self.generate_uniform_matrix(-self.vec_bound, self.vec_bound, size=(self.pop_size, self.problem.n_dims))

        # Initialize personal bests (pBest) and archive (initialized as the non-dominated solutions)
        self.pop_local = self.pop.copy()
        fronts_idx, fronts_new = self.non_dominated_sorting(self.pop.agents)
        self.archive = fronts_new[0]

    def evolve(self, epoch):
        pop_new = []
        for idx in range(0, self.pop_size):
            # Select leader from the archive
            leader = self.select_leader(self.archive)

            # Update velocity
            r1, r2 = self.generator.random(self.problem.n_dims), self.generator.random(self.problem.n_dims)
            vec = (self.inertia_weight * self.velocities[idx] +
                             self.c1 * r1 * (self.pop_local.agents[idx].solution - self.pop.agents[idx].solution) +
                             self.c2 * r2 * (leader.solution - self.pop.agents[idx].solution))
            # Clamp velocities
            vec = np.clip(vec, -self.vec_bound, self.vec_bound)

            # Update position
            pos_new = vec + self.pop.agents[idx].solution
            agent = Agent(solution=pos_new)
            agent = self.repair_agent(agent)
            agent = self.evaluate_agent(agent)
            pop_new.append(agent)

            if self.dominates(agent, self.pop_local.agents[idx]):
                self.pop_local.agents[idx] = agent

        # Update the population
        self.pop = Population(pop_new)

        # Perform non-dominated sorting
        fronts_idx, fronts_new = self.non_dominated_sorting(pop_new + self.pop.agents)
        self.archive = fronts_new[0]
