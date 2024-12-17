#!/usr/bin/env python
# Created by "Thieu" at 3:14 PM, 22/10/2024 --------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from typing import Union, List
import matplotlib.pyplot as plt
import numpy as np
from metamoo.core.prototype import Agent
from metamoo.visualizer.base_drawer import BaseDrawer


class ScatterPlot(BaseDrawer):
    def __init__(self, fig_size=(10, 6), style=None, title=None,
                 show_plot=True, save_file_path=None, file_exts=(".png", ".pdf"),
                 grid=True, view_angle=(30, 30), legend_name="Pareto Front"):
        super().__init__(fig_size, style, title, show_plot, save_file_path, file_exts)
        self.grid = grid
        self.view_angle = view_angle
        self.legend_name = legend_name

    def plot(self, data: Union[List[Agent], np.ndarray], objectives=(1, 2), xyz_labels=None, **kwargs):
        # Format the data
        data = self.check_data(data)
        # Format the selected objectives
        objectives = self.check_objectives(objectives, constraint_dim=True)     # [0, 1]
        # Format the xyz_labels
        xyz_labels = self.check_xyz_labels(xyz_labels, objectives=list(np.array(objectives) + 1))   # [1, 2] for labels

        # Update the front
        front = data[:, objectives]

        fig = plt.figure(figsize=self.fig_size, clear=True)
        if len(objectives) == 2:
            plt.scatter(front[:, 0], front[:, 1], label=self.legend_name, **kwargs)
            if self.title:
                plt.title(self.title)
            if xyz_labels:
                plt.xlabel(xyz_labels[0])
                plt.ylabel(xyz_labels[1])
        elif len(objectives) == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(front[:, 0], front[:, 1], front[:, 2], label=self.legend_name, **kwargs)
            if self.title:
                ax.set_title(self.title)
            if xyz_labels:
                ax.set_xlabel(xyz_labels[0])
                ax.set_ylabel(xyz_labels[1])
                ax.set_zlabel(xyz_labels[2])
            ax.view_init(elev=self.view_angle[0], azim=self.view_angle[1])
        else:
            raise ValueError("Unsupported plot type. Use '2D' or '3D'.")

        if self.grid:
            plt.grid(True)
        plt.legend()
        if self.save_file_path:
            for etx in self.file_exts:
                if etx not in self.save_file_path:
                    plt.savefig(f"{self.save_file_path}.{etx}", bbox_inches='tight')
        if self.show_plot:
            plt.show()
