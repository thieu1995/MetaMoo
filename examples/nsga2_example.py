# Import the necessary classes and functions from the library

from metamoo.algorithms.nsga2 import NSGA2
from metamoo.core.objectives import objective1, objective2
from metamoo.core.problem import Problem, FloatVar
from metamoo.utils.visualization import plot_pareto_front


def run_nsga2_example():

    # Create a multi-objective problem with constraints
    problem = Problem(
        objectives=[objective1, objective2],
        bounds=FloatVar(lb=(-1000., )*10000, ub=(1000.,)*10000),
    )

    # Initialize the NSGA-II algorithm with a population size of 20 and 100 generations
    nsga2 = NSGA2(epoch=50, pop_size=20)
    
    # Run the evolution process to optimize the objectives
    result = nsga2.solve(problem)
    
    # Plot the resulting Pareto front
    plot_pareto_front(result)

# Run the example
if __name__ == "__main__":
    run_nsga2_example()
