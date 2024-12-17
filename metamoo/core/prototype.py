#!/usr/bin/env python
# Created by "Thieu" at 11:24 AM, 10/10/2024 -------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from typing import List, Optional
import numpy as np


class Agent:
    def __init__(self, solution=None, objectives=None, violations=None, feasible=None):
        self.solution = solution        # The solution (variables) representing the Agent
        self.objectives = objectives    # Objective values after evaluation
        self.violations = violations    # Amount of violation for each constraint
        self.feasible = feasible        # Feasibility status (True/False)

    def __repr__(self):
        return f"Agent(vars={self.solution}, objs={self.objectives}, " \
               f"violations={self.violations}, feasible={self.feasible})"

    def to_str(self):
        return f"Agent(objs={self.objectives})"

    def copy(self):
        """Custom copy method to create a shallow copy of the Agent."""
        return Agent(
            solution=None if self.solution is None else self.solution.copy(),
            objectives=None if self.objectives is None else self.objectives.copy(),
            violations=None if self.violations is None else self.violations.copy(),
            feasible=self.feasible
        )


class Population:
    def __init__(self, agents: List[Agent]):
        self.agents = agents

    def get_agents(self) -> List[Agent]:
        return self.agents

    def set_agents(self, agents: List[Agent]):
        self.agents = agents

    def empty_population(self):
        self.agents = []

    def copy(self):
        """Custom copy method to create a new Population object."""
        # Create a new list of agents by calling copy() on each Agent
        new_agents = [agent.copy() for agent in self.agents]
        return Population(new_agents)

    def get_solutions(self):
        return np.array([agent.solution for agent in self.agents])

    def get_objectives(self):
        return np.array([agent.objectives for agent in self.agents])

    # def get_n_best_agents_by_objective(self, n: int, obj_index: int) -> Optional[List[Agent], Agent]:
    #     agents = sorted(self.agents, key=lambda agent: agent.objectives[obj_index])
    #     return agents[:n]
    #
    # def get_best_agents_by_objective(self, objective_index: int) -> List[Agent]:
    #     best_objective_value = min(agent.objectives[objective_index] for agent in self.agents)
    #     return [agent for agent in self.agents if agent.objectives[objective_index] == best_objective_value]
    #
    #
    # def get_best_or_worst_agents_by_objective(self, best: bool = True) -> List[Agent]:
    #     if best:
    #         best_objective_value = min(agent.objectives for agent in self.agents)
    #         return [agent for agent in self.agents if agent.objectives == best_objective_value]
    #     else:
    #         worst_objective_value = max(agent.objectives for agent in self.agents)
    #         return [agent for agent in self.agents if agent.objectives == worst_objective_value]
    #
    # def get_best_or_worst_agents_by_violations(self, best: bool = True) -> List[Agent]:
    #     if best:
    #         least_violations = min(agent.violations for agent in self.agents)
    #         return [agent for agent in self.agents if agent.violations == least_violations]
    #     else:
    #         most_violations = max(agent.violations for agent in self.agents)
    #         return [agent for agent in self.agents if agent.violations == most_violations]
    #
    # def get_feasible_agents(self) -> List[Agent]:
    #     return [agent for agent in self.agents if agent.feasible]


