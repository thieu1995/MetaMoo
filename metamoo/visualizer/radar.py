#!/usr/bin/env python
# Created by "Thieu" at 23:55, 23/10/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from typing import Union, List
import numpy as np
import matplotlib.pyplot as plt
from metamoo.core.prototype import Agent
from metamoo.visualizer.base_drawer import BaseDrawer


class RadarPlot(BaseDrawer):
    def __init__(self, fig_size=(10, 6), style=None, title=None,
                 show_plot=True, save_file_path=None, file_exts=(".png", ".pdf"),
                 fill_area=True, fill_alpha=0.25):
        super().__init__(fig_size, style, title, label_name=None, show_plot=show_plot,
                         save_file_path=save_file_path, file_exts=file_exts)
        self.fill_area = fill_area
        self.fill_alpha = fill_alpha

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

    def plot(self, data: Union[List[Agent], np.ndarray], objectives=(1, 2, 3), obj_names=None,
             legend_names=None, legend_loc_dict=None, **kwargs):
        # Format the data
        data = self.check_data(data)

        # Format the selected objectives
        objectives = self.check_objectives(objectives, constraint_dim=False)
        if objectives is None:
            objectives = list(range(data.shape[1]))

        # Format the xyz_labels
        obj_names = self.check_x_labels(obj_names, objectives=list(np.array(objectives) + 1))

        # Update the front
        front = data[:, objectives]

        # Check legend_names
        if isinstance(legend_names, (tuple, list, np.ndarray)):
            if len(legend_names) != front.shape[0]:
                raise ValueError(f"Length of legend_names should equal to number of solutions.")
        else:
            legend_names = [f"Solution {idx}" for idx in range(1, front.shape[0] + 1)]

        angles = np.linspace(0, 2 * np.pi, len(obj_names), endpoint=False).tolist()
        data_to_plot = np.concatenate((front, front[:, [0]]), axis=1)
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=self.fig_size, subplot_kw=dict(polar=True))
        for idx, row in enumerate(data_to_plot):
            ax.plot(angles, row, linewidth=1, linestyle='solid', label=legend_names[idx], **kwargs)
            if self.fill_area:
                ax.fill(angles, row, alpha=self.fill_alpha)

        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(obj_names)

        if self.title:
            plt.title(self.title)

        if type(legend_loc_dict) is not dict:
            legend_loc_dict = {"loc": "upper right", "bbox_to_anchor": (1.1, 1.1)}

        plt.legend(**legend_loc_dict)

        if self.save_file_path:
            for etx in self.file_exts:
                if etx not in self.save_file_path:
                    plt.savefig(f"{self.save_file_path}.{etx}", bbox_inches='tight')
        if self.show_plot:
            plt.show()

        plt.close(fig)
