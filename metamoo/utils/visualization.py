#!/usr/bin/env python
# Created by "Thieu" at 10:24 AM, 10/10/2024 -------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import matplotlib.pyplot as plt
from metamoo.core.prototype import Population


def plot_pareto_front(pop: Population):
    objectives = [agent.objectives for agent in pop.agents]
    obj1 = [obj[0] for obj in objectives]
    obj2 = [obj[1] for obj in objectives]
    
    plt.scatter(obj1, obj2)
    plt.title('Pareto Front')
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.show()
