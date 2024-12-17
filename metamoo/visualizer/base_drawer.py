#!/usr/bin/env python
# Created by "Thieu" at 3:13 PM, 22/10/2024 --------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from typing import List, Union
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from metamoo.core.prototype import Agent


class BaseDrawer:
    def __init__(self, fig_size=(10, 6), style=None, title=None,
                 show_plot=False, save_file_path=None, file_exts=(".png", ".pdf"), **kwargs):
        self.fig_size = fig_size
        self.style = style
        self.title = title
        self.show_plot = show_plot
        self.save_file_path = save_file_path
        self.file_exts = file_exts
        self.kwargs = kwargs

        if self.style:
            plt.style.use(self.style)

    @staticmethod
    def print_available_styles():
        print(plt.style.available)

    def check_data(self, data):
        if isinstance(data, (np.ndarray, list, tuple)):
            if isinstance(data[0], Agent):
                objs = [agent.objectives for agent in data]
                return np.array(objs)
            if isinstance(data[0], (list, tuple, np.ndarray)):
                return np.array(data)
        raise TypeError("Type of data need to be 2-D matrix (np.ndarray, list, tuple), or list of Agents")

    def check_objectives(self, objectives=None, constraint_dim=True):
        if isinstance(objectives, (list, tuple, np.ndarray)):
            if len(objectives) == 2 or len(objectives) == 3:
                return list(np.array(objectives) - 1)
            else:
                raise ValueError("Number of objectives must be 2 or 3")
        elif objectives is None:
            if constraint_dim:
                return [0, 1]
            return None
        else:
            return [0, 1]

    def check_xyz_labels(self, xyz_labels=None, objectives=None):
        if xyz_labels is None:
            return None
        elif isinstance(xyz_labels, (list, tuple, np.ndarray)):
            return list(xyz_labels)
        else:
            return [f"Objective {idx}" for idx in objectives]

    def plot(self, data: Union[List[Agent], np.ndarray], objectives=(1, 2), **kwargs):
        raise NotImplementedError("Subclass must implement abstract method")
