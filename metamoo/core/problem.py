#!/usr/bin/env python
# Created by "Thieu" at 10:56 AM, 10/10/2024 -------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from typing import Union, List, Tuple, Dict
from metamoo.core.prototype import Agent, Population
from metamoo.core.space import BaseVar, IntegerVar, FloatVar, PermutationVar, StringVar, BinaryVar, BoolVar, MixedSetVar


class Problem:
    SUPPORTED_VARS = (IntegerVar, FloatVar, PermutationVar, StringVar, BinaryVar, BoolVar, MixedSetVar)
    SUPPORTED_ARRAYS = (list, tuple, np.ndarray)

    def __init__(self, objectives, bounds: Union[List, Tuple, np.ndarray, BaseVar], constraints=None, **kwargs):
        self.objectives = objectives  # List of objective functions
        self.constraints = constraints or []  # List of constraint functions (optional)
        self._bounds, self.lb, self.ub = None, None, None
        self.seed = None
        self.name = "P"
        self.n_objs = len(objectives)
        self.n_dims, self.save_population = None, False
        self.__set_keyword_arguments(kwargs)
        self.set_bounds(bounds)

    @property
    def bounds(self):
        return self._bounds

    def set_bounds(self, bounds):
        if isinstance(bounds, BaseVar):
            bounds.seed = self.seed
            self._bounds = [bounds, ]
        elif type(bounds) in self.SUPPORTED_ARRAYS:
            self._bounds = []
            for bound in bounds:
                if isinstance(bound, BaseVar):
                    bound.seed = self.seed
                else:
                    raise ValueError(f"Invalid bounds. All variables in bounds should be an instance of {self.SUPPORTED_VARS}")
                self._bounds.append(bound)
        else:
            raise TypeError(f"Invalid bounds. It should be type of {self.SUPPORTED_ARRAYS} or an instance of {self.SUPPORTED_VARS}")
        self.lb = np.concatenate([bound.lb for bound in self._bounds])
        self.ub = np.concatenate([bound.ub for bound in self._bounds])
        self.n_dims = len(self.lb)

    def set_seed(self, seed: int = None) -> None:
        self.seed = seed
        for idx in range(len(self._bounds)):
            self._bounds[idx].seed = seed

    def __set_keyword_arguments(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_name(self) -> str:
        """
        Returns:
            string: The name of the problem
        """
        return self.name

    def get_class_name(self) -> str:
        """Get class name."""
        return self.__class__.__name__

    @staticmethod
    def encode_solution_with_bounds(x, bounds):
        x_new = []
        for idx, var in enumerate(bounds):
            x_new += list(var.encode(x[idx]))
        return np.array(x_new)

    @staticmethod
    def decode_solution_with_bounds(x, bounds):
        x_new, n_vars = {}, 0
        for idx, var in enumerate(bounds):
            temp = var.decode(x[n_vars:n_vars + var.n_vars])
            if var.n_vars == 1:
                x_new[var.name] = temp[0]
            else:
                x_new[var.name] = temp
            n_vars += var.n_vars
        return x_new

    @staticmethod
    def correct_solution_with_bounds(x: Union[List, Tuple, np.ndarray], bounds: List) -> np.ndarray:
        x_new, n_vars = [], 0
        for idx, var in enumerate(bounds):
            x_new += list(var.correct(x[n_vars:n_vars+var.n_vars]))
            n_vars += var.n_vars
        return np.array(x_new)

    @staticmethod
    def generate_solution_with_bounds(bounds: Union[List, Tuple, np.ndarray], encoded: bool = True) -> Union[List, np.ndarray]:
        x = [var.generate() for var in bounds]
        if encoded:
            return Problem.encode_solution_with_bounds(x, bounds)
        return x

    def encode_solution(self, x: Union[List, tuple, np.ndarray]) -> np.ndarray:
        """
        Encode the real-world solution to optimized solution (real-value solution)

        Args:
            x (Union[List, tuple, np.ndarray]): The real-world solution

        Returns:
            The real-value solution
        """
        return self.encode_solution_with_bounds(x, self.bounds)

    def decode_solution(self, x: np.ndarray) -> Dict:
        """
        Decode the encoded solution to real-world solution

        Args:
            x (np.ndarray): The real-value solution

        Returns:
            The real-world (decoded) solution
        """
        return self.decode_solution_with_bounds(x, self.bounds)

    # def correct_solution(self, x: np.ndarray) -> np.ndarray:
    #     """
    #     Correct the solution to valid bounds
    #
    #     Args:
    #         x (np.ndarray): The real-value solution
    #
    #     Returns:
    #         The corrected solution
    #     """
    #     return self.correct_solution_with_bounds(x, self.bounds)

    def repair_agent(self, agent: Agent) -> Agent:
        agent.solution = self.correct_solution_with_bounds(agent.solution, self.bounds)
        return agent

    def generate_solution(self, encoded: bool = True) -> Union[List, np.ndarray]:
        """
        Generate the solution.

        Args:
            encoded (bool): Encode the solution or not

        Returns:
            the encoded/non-encoded solution for the problem
        """
        return self.generate_solution_with_bounds(self.bounds, encoded)

    def generate_empty_agent(self) -> Agent:
        # Generate random variables from space
        sol = self.generate_solution()
        return Agent(sol)

    def generate_agent(self) -> Agent:
        agent = self.generate_empty_agent()
        agent = self.evaluate_agent(agent)
        return agent

    def generate_population(self, n_agents: int = 100) -> Population:
        # Generate random variables from space
        pop = [self.generate_agent() for _ in range(n_agents)]
        return Population(pop)

    def evaluate_agent(self, agent: Agent) -> Agent:
        # Evaluate the Agent using all objective functions
        agent.objectives = np.array([obj(agent.solution) for obj in self.objectives])

        # Check constraints (returns True if feasible, False if not)
        agent.feasible = all(constraint(agent.solution) for constraint in self.constraints)

        return agent

    # def evaluate(self, agent: Agent, penalty_factor=1.0) -> Agent:
    #     # Evaluate the Agent using all objective functions
    #     agent.objectives = np.array([obj(agent.solution) for obj in self.objectives])
    #
    #     # Calculate constraint violations
    #     violations = [constraint(agent.solution) for constraint in self.constraints]
    #
    #     # Track the violation information (positive value indicates violation amount)
    #     agent.violations = [0 if v >= 0 else -v for v in violations]
    #
    #     # Feasibility check: the Agent is feasible if there are no violations
    #     agent.feasible = all(v == 0 for v in agent.violations)
    #
    #     # Apply penalties based on violation amounts
    #     for i, violation in enumerate(agent.violations):
    #         if violation > 0:
    #             agent.objectives += penalty_factor * violation
    #     return agent
