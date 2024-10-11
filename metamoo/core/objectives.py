#!/usr/bin/env python
# Created by "Thieu" at 10:24 AM, 10/10/2024 -------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

def objective1(variables):
    return np.sum(np.square(variables))

def objective2(variables):
    return np.sum(np.square(variables - 2))
