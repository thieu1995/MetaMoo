#!/usr/bin/env python
# Created by "Thieu" at 09:33, 24/10/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from typing import Union, List
import matplotlib.pyplot as plt
import numpy as np
from metamoo.core.prototype import Agent
from metamoo.visualizer.base_drawer import BaseDrawer


class TradeOffPlot(BaseDrawer):
    def __init__(self, fig_size=(10, 6), style=None, title=None,
                 show_plot=True, save_file_path=None, file_exts=(".png", ".pdf"),
                 grid=True, view_angle=(30, 30), cmap='viridis', alpha=0.8):
        super().__init__(fig_size, style, title, show_plot, save_file_path, file_exts)
        self.grid = grid
        self.view_angle = view_angle
        self.cmap = cmap
        self.alpha = alpha

    def plot(self, data: Union[List[Agent], np.ndarray], objectives=(1, 2, 3), xyz_labels=None, **kwargs):
        # Format the data
        data = self.check_data(data)
        # Format the selected objectives
        objectives = self.check_objectives(objectives, constraint_dim=True)     # [0, 1]
        # Format the xyz_labels
        xyz_labels = self.check_xyz_labels(xyz_labels, objectives=list(np.array(objectives) + 1))   # [1, 2] for labels

        # Update the front
        front = data[:, objectives]

        if len(objectives) != 3:
            raise TypeError("Trade-off surface plot is only supported for 3D.")

        fig = plt.figure(figsize=self.fig_size, clear=True)
        #  Draw tri-surface for Pareto front
        ax = fig.add_subplot(111, projection='3d')
        # Tạo bề mặt trade-off bằng cách nối các điểm của các mục tiêu
        ax.plot_trisurf(front[:,0], front[:,1], front[:,2], cmap=self.cmap, alpha=self.alpha, **kwargs)

        if self.title:
            ax.set_title(self.title)
        if xyz_labels:
            ax.set_xlabel(xyz_labels[0])
            ax.set_ylabel(xyz_labels[1])
            ax.set_zlabel(xyz_labels[2])
        ax.view_init(elev=self.view_angle[0], azim=self.view_angle[1])

        if self.grid:
            plt.grid(True)
        if self.save_file_path:
            for etx in self.file_exts:
                if etx not in self.save_file_path:
                    plt.savefig(f"{self.save_file_path}.{etx}", bbox_inches='tight')
        if self.show_plot:
            plt.show()
