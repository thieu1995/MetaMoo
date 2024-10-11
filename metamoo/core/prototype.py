#!/usr/bin/env python
# Created by "Thieu" at 11:24 AM, 10/10/2024 -------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from typing import List


class Agent:
    def __init__(self, solution):
        self.solution = solution    # The solution (variables) representing the Agent
        self.objectives = None          # Objective values after evaluation
        self.violations = None          # Amount of violation for each constraint
        self.feasible = None            # Feasibility status (True/False)

    def __repr__(self):
        return f"Agent(vars={self.solution}, objs={self.objectives}, " \
               f"violations={self.violations}, feasible={self.feasible})"

class Population:
    def __init__(self, agents: List[Agent]):
        self.agents = agents
