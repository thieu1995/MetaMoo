#!/usr/bin/env python
# Created by "Thieu" at 9:08 AM, 16/10/2024 --------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from typing import List
import time
import numpy as np
from metamoo.core.repairer import BoundRepair
from metamoo.core.prototype import Agent
from metamoo.utils.distance import calculate_crowding_distance
from metamoo.utils import pareto


class Optimizer:
    def __init__(self, seed=None, repairer=None, *args, **kwargs):
        self.seed = seed
        self.generator = np.random.default_rng(self.seed)
        self.repairer = repairer
        self.epoch, self.pop_size = None, None
        self.problem, self.pop = None, None
        self.fronts_indexes, self.fronts_sorted, self.archive = None, None, None
        self.nfe_per_epoch, self.nfe_counter = 0, 0

    def pre_initialization_hook(self):
        """Hook before initialization starts. Subclasses can override."""
        pass

    def initialization(self):
        self.pop = self.problem.generate_population(self.pop_size)
        if self.repairer is None:
            self.repairer = BoundRepair(lb=self.problem.lb, ub=self.problem.ub)

    def post_initialization_hook(self):
        """Hook after initialization completes and before the main loop starts."""
        pass

    def before_evolution(self, epoch):
        """Hook to allow custom logic before each evolution step."""
        pass

    def after_evolution(self, epoch):
        """Hook to allow custom logic after each evolution step."""
        pass

    def custom_callback(self, epoch):
        """Custom callback function for additional behavior within each epoch."""
        pass

    def finalize(self):
        """Finalize and process results at the end of the optimization process."""
        pass

    def solve(self, problem):
        self.problem = problem

        # Pre-initialization hook
        self.pre_initialization_hook()

        self.initialization()

        # Post-initialization hook
        self.post_initialization_hook()

        for epoch in range(1, self.epoch+1):
            time_epoch = time.perf_counter()

            # Before evolution step
            self.before_evolution(epoch)

            self.evolve(epoch)

            # After evolution step
            self.after_evolution(epoch)

            # Update fronts
            self.fronts_indexes, self.fronts_sorted = self.non_dominated_sorting(self.pop.agents)

            # Custom callback within the loop
            self.custom_callback(epoch)

            time_epoch = time.perf_counter() - time_epoch
            self.printer(epoch, self.fronts_sorted, time_epoch)

        # Finalize after all epochs
        self.finalize()

        return self.fronts_sorted

    def evolve(self, epoch):
        """Abstract method to be implemented by subclasses for evolving the population."""
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
        return pareto.non_dominated_sorting(agents=agents)

    @staticmethod
    def find_extreme_points(agents: List[Agent]):
        fronts_indexes, fronts_sorted = pareto.non_dominated_sorting(agents)
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

    def generate_uniform_matrix(self, lb, ub, size):
        matrix = self.generator.uniform(lb, ub, size)
        return matrix

    @staticmethod
    def select_leader(pop):
        """Select a leader (global best) from the archive based on crowding distance."""
        if len(pop) == 0:   # Raise ValueError if archive is empty
            raise ValueError(f"Can not calculate leader for an empty population.")
        distances = calculate_crowding_distance(pop)
        leader_index = np.argmax(distances)
        return pop[leader_index]

    @staticmethod
    def dominates(agent_a, agent_b) -> bool:
        if pareto.dominates(agent_a, agent_b):
            return True
        return False
