#!/usr/bin/env python
# Created by "Thieu" at 9:08 AM, 16/10/2024 --------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from typing import List
import time
import numpy as np
from metamoo.core.prototype import Agent
from metamoo.utils.pareto import non_dominated_sorting


class Optimizer:
    def __init__(self, seed=None, repairer=None, *args, **kwargs):
        self.seed = seed
        self.generator = np.random.default_rng(self.seed)
        self.repairer = repairer
        self.epoch, self.pop_size = None, None
        self.problem, self.pop, self.fronts = None, None, None
        self.nfe_per_epoch, self.nfe_counter = 0, 0

    def initialization(self):
        self.pop = self.problem.generate_population(self.pop_size)
        # Perform non-dominated sorting
        self.fronts = self.non_dominated_sorting(self.pop.agents, return_index=True)

    def solve(self, problem):
        self.problem = problem
        self.initialization()

        for epoch in range(1, self.epoch+1):
            time_epoch = time.perf_counter()

            self.evolve(epoch)

            # Update fronts
            self.fronts = self.non_dominated_sorting(self.pop.agents, return_index=True)

            time_epoch = time.perf_counter() - time_epoch
            self.printer(epoch, self.fronts, time_epoch)

        return self.fronts

    def evolve(self, epoch):
        pass

    def callback(self):
        pass

    def printer(self, epoch, fronts, time_epoch=None, diversity=None):
        print(f"Epoch: {epoch}, Best front size: {len(fronts[0])}, "
              f"1st best non-dominated solution: {self.pop[fronts[0][0]].objectives}, "
              f"Time: {time_epoch:.3f} seconds, Diversity: {diversity}.")

    @staticmethod
    def non_dominated_sorting(agents: List[Agent], return_index: bool = True):
        return non_dominated_sorting(agents=agents, return_index=return_index)

    @staticmethod
    def find_extreme_points(agents: List[Agent]):
        fronts, _ = non_dominated_sorting(agents)
        return fronts[0]

    def repair_agent(self, agent: Agent) -> Agent:
        """
        This function is based on optimizer's strategy and problem-specific condition
        DO NOT override this function

        Args:
            agent: The search agent

        Returns:
            The agent with correct solution that can be used to calculate objectives
        """
        agent = self.repairer.do(agent)
        return self.problem.repair_agent(agent)

    def evaluate_agent(self, agent: Agent, counted: bool = True) -> Agent:
        if counted:
            self.nfe_counter += 1
        return self.problem.evaluate_agent(agent)
