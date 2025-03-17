import os
import numpy as np


def _one_hot_encode(indices, class_num):
    """
    Convert an array of indices to one-hot encoded format.

    Parameters:
    indices (array-like): list of index.
    class_num (int): The number of classes.

    Returns:
    numpy.ndarray: relu_one-hot vectors
    """
    size = len(indices)
    one_hot = np.zeros((size, class_num))
    one_hot[np.arange(size), indices] = 1
    return one_hot


def _contain_ignore(target_transform):
    if target_transform:
        from .target_transforms import Ignore, Compose

        if isinstance(target_transform, Ignore):
            raise ValueError('target_transforms.Ignore must be passed to "ignore" argument.')

        if isinstance(target_transform, Compose):
            for t in target_transform.target_transforms:
                if isinstance(t, Ignore):
                    raise ValueError(
                        'target_transforms.Ignore must be passed to "ignore" argument.'
                    )

    return target_transform


def _check_ins(name, val, cls, allow_none=False):
    """
    Check if a value is an instance of a specified class.

    Args:
        name (str): The name of the argument being checked.
        val: The value to check.
        cls (type): The class or tuple of classes to check against.
        allow_none (bool, optional): If True, allows `val` to be None. Defaults to False.

    Returns:
        The original value if it passes the check.

    Raises:
        ValueError: If `val` is not an instance of `cls` and `allow_none` is False.
    """
    if allow_none and val is None:
        return val

    if not isinstance(val, cls):
        raise ValueError(
            'Argument "{}" must be {}, but got {}.'.format(
                name, cls.__name__, type(val).__name__
            )
        )
    return val


DATA_ROOT = os.path.join(os.path.expanduser("~"), "data")
