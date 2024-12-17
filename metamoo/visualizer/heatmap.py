#!/usr/bin/env python
# Created by "Thieu" at 19:24, 22/10/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from typing import Union, List
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from metamoo.core.prototype import Agent
from metamoo.visualizer.base_drawer import BaseDrawer


class HeatmapPlot(BaseDrawer):
    def __init__(self, fig_size=(10, 6), style=None, title=None,
                 show_plot=True, save_file_path=None, file_exts=(".png", ".pdf"),
                 annot=True, cmap='coolwarm', line_widths=0.5):
        super().__init__(fig_size, style, title, show_plot=show_plot,
                         save_file_path=save_file_path, file_exts=file_exts)
        self.annot = annot
        self.cmap = cmap
        self.line_widths = line_widths

    def check_x_labels(self, x_labels, objectives=None):
        if isinstance(x_labels, (list, tuple, np.ndarray)):
            if len(x_labels) == len(objectives):
                return x_labels
            else:
                raise ValueError(f"Length of x_labels should equal to number of selected objectives.")
        else:
            return [f"Objective {idx}" for idx in objectives]

    def plot(self, data: Union[List[Agent], np.ndarray], objectives=(1, 2), x_labels=None, y_label=None, **kwargs):
        # Format the data
        data = self.check_data(data)

        # Format the selected objectives
        objectives = self.check_objectives(objectives, constraint_dim=False)
        if objectives is None:
            objectives = list(range(data.shape[1]))

        # Format the xyz_labels
        x_labels = self.check_x_labels(x_labels, objectives=list(np.array(objectives) + 1))

        # Update the front
        front = data[:, objectives]

        # Create one figure object
        fig, ax = plt.subplots(figsize=self.fig_size)

        df = pd.DataFrame(front, columns=x_labels)
        sns.heatmap(df, annot=self.annot, cmap=self.cmap, linewidths=self.line_widths, ax=ax, **kwargs)

        if self.title:
            plt.title(self.title)
        if y_label:
            ax.set_ylabel(y_label)

        if self.save_file_path:
            for etx in self.file_exts:
                if etx not in self.save_file_path:
                    plt.savefig(f"{self.save_file_path}.{etx}", bbox_inches='tight')
        if self.show_plot:
            plt.show()

        plt.close(fig)
