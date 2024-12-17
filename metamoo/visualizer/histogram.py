#!/usr/bin/env python
# Created by "Thieu" at 00:48, 24/10/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from typing import Union, List
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from metamoo.core.prototype import Agent
from metamoo.visualizer.base_drawer import BaseDrawer


class HistogramPlot(BaseDrawer):
    def __init__(self, fig_size=(5, 6), style=None, title=None,
                 show_plot=True, save_file_path=None, file_exts=(".png", ".pdf"),
                 list_bins=5, list_colors='skyblue'):
        super().__init__(fig_size, style, title, show_plot=show_plot,
                         save_file_path=save_file_path, file_exts=file_exts)
        self.list_bins = list_bins
        self.list_colors = list_colors

    def check_list_values(self, list_values, n_objs, default_value=None):
        if isinstance(list_values, (list, tuple, np.ndarray)):
            if len(list_values) == n_objs:
                return list_values
            else:
                return [list_values[0], ] * n_objs
        elif isinstance(list_values, (int, float)):
            return [int(list_values), ] * n_objs
        else:
            return [default_value, ] * n_objs

    def check_x_labels(self, x_labels, objectives=None):
        if isinstance(x_labels, (list, tuple, np.ndarray)):
            if len(x_labels) == len(objectives):
                return x_labels
            else:
                raise ValueError(f"Length of x_labels should equal to number of selected objectives.")
        else:
            return [f"Objective {idx}" for idx in objectives]

    def plot(self, data: Union[List[Agent], np.ndarray], objectives=(1, 2), obj_names=None, **kwargs):
        # Format the data
        data = self.check_data(data)

        # Format the selected objectives
        objectives = self.check_objectives(objectives, constraint_dim=False)
        if objectives is None:
            objectives = list(range(data.shape[1]))

        # Format the x_labels
        x_labels = self.check_x_labels(obj_names, objectives=list(np.array(objectives) + 1))

        # Update the front
        front = data[:, objectives]

        # Check list_bins
        list_bins = self.check_list_values(self.list_bins, len(objectives), default_value=5)
        list_colors = self.check_list_values(self.list_colors, len(objectives), default_value='skyblue')

        # Create one figure object
        fig, axes = plt.subplots(len(objectives), 1, figsize=self.fig_size, constrained_layout=True)
        if len(objectives) == 1:
            axes = [axes]
        for idx, obj in enumerate(objectives):
            sns.histplot(front[:, idx], bins=list_bins[idx], color=list_colors[idx], ax=axes[idx], **kwargs)
            axes[idx].set_xlabel(x_labels[idx])

        if self.title:
            fig.suptitle(self.title)

        if self.save_file_path:
            for ext in self.file_exts:
                plt.savefig(f"{self.save_file_path}.{ext}", bbox_inches='tight')

        if self.show_plot:
            plt.show()
        plt.close(fig)
