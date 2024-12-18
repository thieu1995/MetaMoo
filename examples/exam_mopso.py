#!/usr/bin/env python
# Created by "Thieu" at 18:02, 17/12/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from metamoo import Problem, FloatVar
from metamoo import MOPSO
from metamoo import ScatterPlot


def objective_function1(X):
    return np.sum(X ** 2)

def objective_function2(X):
    return np.sum((X - 1) ** 2 - np.abs(X))

def objective_function3(X):
    return np.sum(np.sin(X) + X**2 - 5*X)


def run_model():
    # Create a multi-objective problem
    problem = Problem(
        objectives=[objective_function1, objective_function2, objective_function3],
        bounds=FloatVar(lb=(-1.,) * 15, ub=(1.,) * 15)
    )
    SEED = 10

    # Initialize the MO-PSO algorithm
    model = MOPSO(epoch=100, pop_size=50, inertia_weight=0.5, c1=1.5, c2=1.5, seed=SEED)

    # Run the evolution process to optimize the objectives
    model.solve(problem)

    ## Plot the resulting Pareto front
    plotter = ScatterPlot(fig_size=(10, 6), style=None, title="Pareto front - Scatter",
                          show_plot=True, save_file_path=None, file_exts=(".png", ".pdf"),
                          grid=True, view_angle=(30, 30), legend_name='Pareto Front')

    # Plot 2D for best pareto front found
    plotter.plot(model.fronts_sorted[0], objectives=[1, 2], xyz_labels=None, c="r", marker="o")

    # Plot 3D for best pareto front found
    plotter.plot(model.fronts_sorted[0], objectives=[1, 2, 3], xyz_labels=["obj 1", "obj 2", "obj 3"])


# Run the example
if __name__ == "__main__":
    run_model()
