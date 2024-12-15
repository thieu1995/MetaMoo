#!/usr/bin/env python
# Created by "Thieu" at 16:49, 15/12/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from metamoo.utils.ref_point import generate_reference_points, associate_with_reference_points


# Reference point generation
def generate_reference_points2(n_objs, n_divisions):
    if n_objs == 1:
        return np.linspace(0, 1, n_divisions).reshape(-1, 1)
    else:
        ref_points = []
        for i in range(n_divisions + 1):
            sub_points = generate_reference_points2(n_objs - 1, n_divisions - i)
            for sub_point in sub_points:
                ref_points.append([i / n_divisions] + list(sub_point))
        return np.array(ref_points)


if __name__ == "__main__":

    ref_points2 = generate_reference_points2(3, 6)
    print(ref_points2)
    print(ref_points2.shape)

    ref_points = generate_reference_points(3, 6)
    print(ref_points)
    print(ref_points.shape)

    pop_objs = np.array([[1, 2, 3],
                           [2, 1.5, 4],
                           [2.5, 1, 3.5],
                           [3, 3, 5],
                           [3.5, 2.5, 2],
                           [2, 3, 4],
                           [4, 3, 5],
                           [1.5, 2.4, 3.4]])

    c1, d1 = associate_with_reference_points(pop_objs, ref_points)
    print(c1)
    print(d1)
