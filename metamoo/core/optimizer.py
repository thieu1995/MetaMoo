#!/usr/bin/env python
# Created by "Thieu" at 9:08 AM, 16/10/2024 --------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from typing import List
import time
import numpy as np
from metamoo.core.prototype import Agent
from metamoo.utils.pareto import non_dominated_sorting as nds


class Optimizer:
    def __init__(self, seed=None, repairer=None, *args, **kwargs):
        self.seed = seed
        self.generator = np.random.default_rng(self.seed)
        self.repairer = repairer
        self.epoch, self.pop_size = None, None
        self.problem, self.pop = None, None
        self.fronts_indexes, self.fronts_sorted = None, None
        self.nfe_per_epoch, self.nfe_counter = 0, 0

    def initialization(self):
        self.pop = self.problem.generate_population(self.pop_size)
        # Perform non-dominated sorting
        self.fronts_indexes, self.fronts_sorted = self.non_dominated_sorting(self.pop.agents)

    def solve(self, problem):
        self.problem = problem
        self.initialization()

        for epoch in range(1, self.epoch+1):
            time_epoch = time.perf_counter()

            self.evolve(epoch)

            # Update fronts
            self.fronts_indexes, self.fronts_sorted = self.non_dominated_sorting(self.pop.agents)

            time_epoch = time.perf_counter() - time_epoch
            self.printer(epoch, self.fronts_sorted, time_epoch)

        return self.fronts_sorted

    def evolve(self, epoch):
        pass

    def callback(self):
        pass

    def printer(self, epoch, fronts, time_epoch=None, diversity=None, print_best_front=True):
        if print_best_front:
            bf = "Best front: \n\t" + "\n\t".join([agent.to_str() for agent in fronts[0]])
        else:
            bf = ""
        print(f"Epoch: {epoch}, Best front size: {len(fronts[0])}, "
              f"Time: {time_epoch:.4f} seconds, Diversity: {diversity}, {bf}")

    @staticmethod
    def non_dominated_sorting(agents: List[Agent]):
        return nds(agents=agents)

    @staticmethod
    def find_extreme_points(agents: List[Agent]):
        fronts_indexes, fronts_sorted = nds(agents)
        return fronts_sorted[0]

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
