#!/usr/bin/env python
# Created by "Thieu" at 4:28 PM, 16/10/2024 -------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from metamoo import Problem, FloatVar
from metamoo import NSGA, NsgaSelector, ArithmeticCrossover, SwapMutator, BoundRepair
from metamoo import ScatterPlot


def objective_function1(X):
    return np.sum(X ** 2)

def objective_function2(X):
    return np.sum((X - 1) ** 2 - np.abs(X))

def objective_function3(X):
    return np.sum(np.sin(X) + X**2 - 5*X)


def run_nsga():
    # Create a multi-objective problem
    problem = Problem(
        objectives=[objective_function1, objective_function2, objective_function3],
        bounds=FloatVar(lb=(-1.,) * 15, ub=(1.,) * 15)
    )
    SEED = 10

    # Initialize the NSGA-II algorithm
    model = NSGA(epoch=50, pop_size=20,
                 selector=NsgaSelector(seed=SEED),
                 crossover=ArithmeticCrossover(crossover_rate=0.8, seed=SEED),
                 mutator=SwapMutator(mutation_rate=0.1, seed=SEED),
                 repairer=BoundRepair(problem.lb, problem.ub),
                 seed=SEED)

    # Run the evolution process to optimize the objectives
    model.solve(problem)

    # # Plot the resulting Pareto front
    # Plot 2D for best pareto front found
    scatter_2d = ScatterPlot(model.fronts_sorted[0], plot_type='2D', title='2D Scatter Plot',
                             xlabel='Objective 1', ylabel='Objective 2', objectives=[1, 2])
    scatter_2d.plot()

    # Plot 3D for best pareto front found
    scatter_3d = ScatterPlot(model.fronts_sorted[0], plot_type='3D', title='3D Scatter Plot',
                             xlabel='Objective 1', ylabel='Objective 2', zlabel='Objective 3',
                             view_angle=(30, 45), objectives=[1, 2, 3])
    scatter_3d.plot()


# Run the example
if __name__ == "__main__":
    run_nsga()
