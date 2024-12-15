#!/usr/bin/env python
# Created by "Thieu" at 02:01, 15/12/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np


# Generate reference points
def generate_reference_points(n_objs, n_divisions):
    ref_points = []
    for point in np.ndindex(*(n_divisions + 1 for _ in range(n_objs))):
        if sum(point) <= n_divisions:
            ref_points.append(np.array(point) / n_divisions)
    return np.array(ref_points)


# Associate solutions to reference points
def associate_with_reference_points(pop_objs, ref_points):
    normalized_objs = pop_objs / np.linalg.norm(pop_objs, axis=1, keepdims=True)
    distances = np.linalg.norm(normalized_objs[:, np.newaxis, :] - ref_points[np.newaxis, :, :], axis=2)
    closest_points = np.argmin(distances, axis=1)
    return closest_points, distances[np.arange(len(distances)), closest_points]
