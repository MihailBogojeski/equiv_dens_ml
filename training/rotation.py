import scipy
import numpy as np


def random_rotation_matrix():
    """
    Generates a random 3D rotation matrix from axis and angle.
    Args:
        numpy_random_state: numpy random state object
    Returns:
        Random rotation matrix.
    """

    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis) + 1e-8
    theta = 2 * np.pi * np.random.uniform(0.0, 1.0)
    return rotation_matrix(axis, theta)


def rotation_matrix(axis, theta):
    return scipy.linalg.expm(np.cross(np.eye(3), axis * theta))
