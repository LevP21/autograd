from typing import Tuple

import numpy as np

from tensor import Tensor

def unbroadcast(
    grad: Tensor,
    shape: Tuple[int, ...]
):
    """
    Squeezes the shape of the input tensor to the specific shape

    Args:
        grad (Tensor): The input tensor
        shape: The shape to squeeze the input tensor to

    Returns:
        Tensor: Tensor with the squeezed value
    """

    if grad.shape == ():
        grad = np.full(shape, grad)

    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)

    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


def element_counts_along_axis(
    array: np.ndarray,
    axis: int
) -> np.ndarray:
    """
    Counts every element of the array along the particular axis
    
    Args:
        array (np.ndarray): Input array
        axis (int): Axis along which to count elements

    Returns:
        np.ndarray: Array with the shape of the input array with every element set as count of a value with the same index in the input array
    """

    moved = np.moveaxis(array, axis, -1)

    result = np.empty(moved.shape, dtype=np.intp)

    for idx in np.ndindex(moved.shape[:-1]):
        row = moved[idx]
        _, inverse, counts = np.unique(
            row, return_inverse=True, return_counts=True
        )
        result[idx] = counts[inverse]

    return np.moveaxis(result, -1, axis)


def has_duplicates_along_axis(
    array: np.ndarray,
    axis: int
) -> bool:
    """
    Checks if there're duplicate elements in the input array along the particular axis

    Args:
        array (np.ndarray): Input array
        axis (int): Axis along which to search for duplicates

    Returns:
        bool: If there're any duplicates along the axis
    """

    array = np.moveaxis(np.asarray(array), axis, -1)
    array = np.sort(array, axis=-1)

    return np.any(np.diff(array, axis=-1) == 0)