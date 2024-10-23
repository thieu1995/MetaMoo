#!/usr/bin/env python
# Created by "Thieu" at 01:26, 24/10/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from metamoo import HistogramPlot


# Example usage
pareto_front = np.array([
    [0.1, 1.8, 0.2, 1.4, 0.5],
    [0.2, 2.6, 0.3, 3.5, 0.7],
    [0.3, 3.5, 0.4, 2.3, 0.6],
    [0.4, 5.3, 0.5, 1.2, 0.8],
    [0.5, 7.2, 0.6, 2.1, 0.9],
    [0.6, 2.4, 0.1, 1.8, 0.4],
    [0.7, 4.3, 0.7, 1.6, 0.2]
])

# Create instance of HeatmapPlot
plotter = HistogramPlot(fig_size=(10, 6), style=None, title="Pareto front - Heatmap",
                        show_plot=True, save_file_path=None, file_exts=(".png", ".pdf"),
                        list_bins=None, list_colors=None)

plotter.plot(pareto_front, objectives=[1, 4], obj_names=None)

plotter.plot(pareto_front, objectives=[1, 3, 4], obj_names=["Obj 1", "Obj 3", "Obj 4"])

plotter.plot(pareto_front, objectives=None, obj_names=None)
