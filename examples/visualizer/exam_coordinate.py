#!/usr/bin/env python
# Created by "Thieu" at 12:15, 24/10/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from metamoo import ParallelCoordinatePlot


# Example usage
pareto_front = np.array([
    [0.1, 0.8, 0.2, 0.4, 0.5],
    [0.2, 0.6, 0.3, 0.5, 0.7],
    [0.3, 0.5, 0.4, 0.3, 0.6],
    [0.4, 0.3, 0.5, 0.2, 0.8],
    [0.5, 0.2, 0.6, 0.1, 0.9],
    [0.6, 0.4, 0.1, 0.8, 0.4],
    [0.7, 0.3, 0.7, 0.6, 0.2]
])

# Create instance of HeatmapPlot
plotter = ParallelCoordinatePlot(fig_size=(10, 6), style=None, title="Pareto front - ParallelCoordinatePlot",
                      show_plot=True, save_file_path=None, file_exts=(".png", ".pdf"),)

# objectives=(1, 2), x_labels=None,
#              legend_names=None, axvline_param=None

plotter.plot(pareto_front, objectives=[1, 2, 3, 4], x_labels=["Obj 1", "Obj 2", "Obj 3", "Obj 4"])

plotter.plot(pareto_front, objectives=None, x_labels=None)
