#!/usr/bin/env python
# Created by "Thieu" at 10:49, 24/10/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from typing import Union, List
import matplotlib.pyplot as plt
import numpy as np
from metamoo.core.prototype import Agent
from metamoo.visualizer.base_drawer import BaseDrawer


class SurfacePlot(BaseDrawer):
    def __init__(self, fig_size=(10, 6), style=None, title=None,
                 show_plot=True, save_file_path=None, file_exts=(".png", ".pdf"),
                 grid=True, view_angle=(30, 30), cmap='viridis', alpha=0.8):
        super().__init__(fig_size, style, title, show_plot, save_file_path, file_exts)
        self.grid = grid
        self.view_angle = view_angle
        self.cmap = cmap
        self.alpha = alpha

    def check_xyz_labels(self, xyz_labels=None, objectives=None):
        if xyz_labels is None:
            return None
        elif isinstance(xyz_labels, (list, tuple, np.ndarray)):
            return list(xyz_labels)
        else:
            return [f"Objective {idx}" for idx in objectives] + ["z value"]

    def plot(self, data: Union[List[Agent], np.ndarray], z_func=None, objectives=(1, 2),
             xyz_labels=None, n_z_point=100, show_contour=True, show_colorbar=True, **kwargs):
        # Format the data
        data = self.check_data(data)
        # Format the selected objectives
        objectives = self.check_objectives(objectives, constraint_dim=True)     # [0, 1]
        # Format the xyz_labels
        xyz_labels = self.check_xyz_labels(xyz_labels, objectives=list(np.array(objectives) + 1))   # [1, 2] for labels

        # Update the front
        front = data[:, objectives]

        if len(objectives) != 2:
            raise TypeError("Surface plot is only supported for 2D objectives and z functions.")

        # Check number of z points
        if type(z_func) is not int:
            n_z_point = 100

        # Create a grid of points for contour plotting
        x = np.linspace(np.min(front[:,0]), np.max(front[:,1]), n_z_point)
        y = np.linspace(np.min(front[:,0]), np.max(front[:,1]), n_z_point)
        X, Y = np.meshgrid(x, y)

        # Create a third variable based on some function of the two objectives
        Z = z_func(X, Y)
        # Create a 3D plot
        fig = plt.figure(figsize=self.fig_size, clear=True)
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        surf = ax.plot_surface(X, Y, Z, cmap=self.cmap, alpha=self.alpha, **kwargs)

        # Add contour lines
        if show_contour:
            contour = ax.contour(X, Y, Z, zdir='z', offset=np.min(Z), cmap=self.cmap, linewidths=0.5, **kwargs)

        if self.title:
            ax.set_title(self.title)
        if xyz_labels:
            ax.set_xlabel(xyz_labels[0])
            ax.set_ylabel(xyz_labels[1])
            ax.set_zlabel(xyz_labels[2])
        ax.view_init(elev=self.view_angle[0], azim=self.view_angle[1])

        # Show colorbar
        if show_colorbar:
            plt.colorbar(surf)  # Show color scale

        if self.grid:
            plt.grid(True)
        if self.save_file_path:
            for etx in self.file_exts:
                if etx not in self.save_file_path:
                    plt.savefig(f"{self.save_file_path}.{etx}", bbox_inches='tight')
        if self.show_plot:
            plt.show()
