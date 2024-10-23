#!/usr/bin/env python
# Created by "Thieu" at 21:33, 23/10/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from metamoo import HeatmapPlot


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
plotter = HeatmapPlot(fig_size=(10, 6), style=None, title="Pareto front - Heatmap",
                      show_plot=True, save_file_path=None, file_exts=(".png", ".pdf"),
                      annot=True, cmap='coolwarm', line_widths=0.5)

plotter.plot(pareto_front, objectives=[1, 4], x_labels=None, y_label="PF solution")

plotter.plot(pareto_front, objectives=[1, 3, 4], x_labels=["Obj 1", "Obj 3", "Obj 4"], y_label=None)

plotter.plot(pareto_front, objectives=None, y_label="PF solution")
