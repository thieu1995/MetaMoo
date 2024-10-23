#!/usr/bin/env python
# Created by "Thieu" at 18:45, 22/10/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from metamoo import ScatterPlot


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

plotter = ScatterPlot(fig_size=(10, 6), style=None, title="Pareto front - Scatter",
                      label_name='Pareto Front', show_plot=True, save_file_path=None,
                      file_exts=(".png", ".pdf"), grid=True, view_angle=(30, 30))

plotter.plot(pareto_front, objectives=[1, 4], xyz_labels=None)
plotter.plot(pareto_front, objectives=[2, 5], xyz_labels=["obj 2", "obj 5"])

plotter.plot(pareto_front, objectives=[2, 3, 4], xyz_labels=["obj 2", "obj 3", "obj 4"])
plotter.plot(pareto_front, objectives=[1, 2, 5], xyz_labels="default")
