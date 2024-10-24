#!/usr/bin/env python
# Created by "Thieu" at 01:44, 24/10/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from typing import Union, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metamoo import Agent
from metamoo.visualizer.base_drawer import BaseDrawer


class ParallelCoordinatePlot(BaseDrawer):
    def __init__(self, fig_size=(10, 6), style=None, title=None,
                 show_plot=True, save_file_path=None, file_exts=(".png", ".pdf"),
                 color='blue', alpha=0.5):
        super().__init__(fig_size, style, title, show_plot=show_plot,
                         save_file_path=save_file_path, file_exts=file_exts)
        self.color = color
        self.alpha = alpha
        self.class_column = None

    def check_x_labels(self, x_labels, objectives=None):
        if isinstance(x_labels, (list, tuple, np.ndarray)):
            if len(x_labels) == len(objectives):
                return x_labels
            else:
                raise ValueError(f"Length of x_labels should equal to number of selected objectives.")
        else:
            return [f"Objective {idx}" for idx in objectives]

    def check_objectives(self, objectives=None, constraint_dim=True):
        if isinstance(objectives, (list, tuple, np.ndarray)):
            return list(np.array(objectives) - 1)
        else:
            return None

    def normalize_data(self, data):
        min_vals = data.min(axis=0)
        max_vals = data.max(axis=0)
        return (data - min_vals) / (max_vals - min_vals)

    def plot(self, data: Union[List[Agent], np.ndarray], objectives=(1, 2), x_labels=None,
             legend_names=None, axvline_param=None, **kwargs):
        # Format the data
        data = self.check_data(data)
        # Format the selected objectives
        objectives = self.check_objectives(objectives, constraint_dim=False)
        if objectives is None:
            objectives = list(range(data.shape[1]))
        # Format the x_labels
        x_labels = self.check_x_labels(x_labels, objectives=list(np.array(objectives) + 1))

        # Check legend_names
        if isinstance(legend_names, (tuple, list, np.ndarray)):
            if len(legend_names) != data.shape[0]:
                raise ValueError(f"Length of legend_names should equal to number of solutions.")
        else:
            legend_names = [f"Solution {idx}" for idx in range(1, data.shape[0] + 1)]

        # Update the front
        front = data[:, objectives]
        front = self.normalize_data(front)

        # Subset the data according to the objectives
        df = pd.DataFrame(front, columns=x_labels)

        # Plot
        fig, ax = plt.subplots(figsize=self.fig_size)
        color_cycle = plt.cm.tab10(np.linspace(0, 1, len(df)))

        for idx, row in df.iterrows():
            ax.plot(df.columns, row, color=color_cycle[idx], alpha=self.alpha, label=legend_names[idx], **kwargs)

        # Draw vertical lines and add labels with margin
        if axvline_param is not dict:
            axvline_param = {"color": "grey", "linestyle": "--", "linewidth": 1}
        label_margin = 0.05
        for x in df.columns:
            ax.axvline(x=x, **axvline_param)
            ax.text(x, -label_margin, '0', verticalalignment='bottom', horizontalalignment='center',
                    color='black', fontsize=11, transform=ax.get_xaxis_transform())
            ax.text(x, 1 + label_margin, '1', verticalalignment='top', horizontalalignment='center',
                    color='black', fontsize=11, transform=ax.get_xaxis_transform())

        ax.set_ylim(0, 1)
        ax.legend(loc='center', bbox_to_anchor=(1.05, 0.85))

        # Adjust the x-axis text position and title position
        ax.tick_params(axis='x', pad=20)
        if self.title:
            plt.title(self.title, pad=40)

        if self.save_file_path:
            for ext in self.file_exts:
                plt.savefig(f"{self.save_file_path}.{ext}", bbox_inches='tight')

        if self.show_plot:
            plt.show()
        plt.close()
