#!/usr/bin/env python
# Created by "Thieu" at 23:49, 23/10/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from metamoo import RadarPlot


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
plotter = RadarPlot(fig_size=(10, 6), style=None, title="Pareto front - Radar/Spider chart",
                    show_plot=True, save_file_path=None, file_exts=(".png", ".pdf"),
                    fill_area=False, fill_alpha=0.25)

# objectives=(1, 2, 3), obj_names=None,
#              legend_names=None, legend_loc_dict=None

plotter.plot(pareto_front, objectives=[1, 2, 3], obj_names=None, legend_names=None, legend_loc_dict=None)

plotter.plot(pareto_front, objectives=[1, 3, 4, 5], obj_names=["Obj 1", "Obj 3", "Obj 4", "Obj 5"],
             legend_names=None,
             legend_loc_dict={"loc": "upper right", "bbox_to_anchor": (1.1, 1.1), "ncol": 1, "borderaxespad": 0.0})

plotter.plot(pareto_front, objectives=None, obj_names=None, legend_names=None, legend_loc_dict=None)
