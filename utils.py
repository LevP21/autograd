import numpy as np

from tensor import Tensor

def unbroadcast(grad: Tensor, shape):
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