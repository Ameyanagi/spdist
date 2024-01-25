""" Collection of functions to calculate the distance between two curves using spdist metric.

The main fucntions are:
- `spdist`: calculate the distance between two curves using spdist metric.
- `spdist_vector`: calculate the distance between two curves using spdist metric and return a vector of distances.
- `squared_spdist`: calculate the squared distance between two curves using squared spdist metric.
- `squared_spdist_vector`: calculate the squared distance between two curves using squared spdist metric and return a vector of distances.
"""

import numpy as np

def spdist(x: np.ndarray, y: np.ndarray, x_ref: np.ndarray, y_ref: np.ndarray) -> float:
    """Calculate the distance between two curves using spdist metric.

    The distance is calculated by finding the minimum distance between each point in the first curve and the reference curve.
    The distance is only calculated correctly if the `x_ref` is sorted.
    Please refer to [wikipedia](https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points) for more information of the distance calculation.


    Args:
        x (np.ndarray): array of x coordinates of curve 1
        y (np.ndarray): array of y coordinates of curve 1
        x_ref (np.ndarray): array of x coordinates of reference curve
        y_ref (np.ndarray): array of y coordinates of reference curve

    Returns:
        float: minium distance between two curves using spdist metric.
    """

def spdist_vector(
    x: np.ndarray, y: np.ndarray, x_ref: np.ndarray, y_ref: np.ndarray
) -> np.ndarray:
    """Calculate the distance between two curves using spdist metric and return a vector of distances.
    The distance is calculated by finding the minimum distance between each point in the first curve and the reference curve.
    The distance is only calculated correctly if the `x_ref` is sorted.
    Please refer to [wikipedia](https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points) for more information of the distance calculation.

    Args:
        x (np.ndarray): array of x coordinates of curve 1
        y (np.ndarray): array of y coordinates of curve 1
        x_ref (np.ndarray): array of x coordinates of reference curve
        y_ref (np.ndarray): array of y coordinates of reference curve

    Returns:
        np.ndarray: vector of distances between two curves using spdist metric.
    """

def squared_spdist(
    x: np.ndarray, y: np.ndarray, x_ref: np.ndarray, y_ref: np.ndarray
) -> float:
    """Calculate the squared distance between two curves using squared spdist metric.
    The distance is calculated by finding the minimum squared distance between each point in the first curve and the reference curve.
    The distance is only calculated correctly if the `x_ref` is sorted.
    Please refer to [wikipedia](https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points) for more information of the distance calculation. The squared distance is calculated by squaring the distance.

    Args:
        x (np.ndarray): array of x coordinates of curve 1
        y (np.ndarray): array of y coordinates of curve 1
        x_ref (np.ndarray): array of x coordinates of reference curve
        y_ref (np.ndarray): array of y coordinates of reference curve

    Returns:
        float: minium distance between two curves using squared spdist metric.
    """

def squared_spdist_vector(
    x: np.ndarray, y: np.ndarray, x_ref: np.ndarray, y_ref: np.ndarray
) -> np.ndarray:
    """Calculate the squared distance between two curves using squared spdist metric and return a vector of distances.
    The distance is calculated by finding the minimum squared distance between each point in the first curve and the reference curve.
    The distance is only calculated correctly if the `x_ref` is sorted.
    Please refer to [wikipedia](https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points) for more information of the distance calculation. The squared distance is calculated by squaring the distance.

    Args:
        x (np.ndarray): array of x coordinates of curve 1
        y (np.ndarray): array of y coordinates of curve 1
        x_ref (np.ndarray): array of x coordinates of reference curve
        y_ref (np.ndarray): array of y coordinates of reference curve

    Returns:
        np.ndarray: vector of distances between two curves using squared spdist metric.
    """
