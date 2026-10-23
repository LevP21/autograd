from typing import Tuple, Any, List

import numpy as np

from tensor import Tensor
from utils import *

def eg(
    self: Tensor,
    other: Any
) -> Tensor:
    """
    Operator of equality between two tensors

    Args:
        self (Tensor): First input tensor
        other (Any): Second input tensor or other object that can be compared with a tensor

    Returns:
        Tensor: Result of equality operation
    """

    other = other if isinstance(other, Tensor) else Tensor(other)

    return Tensor(self.data == other.data, requires_grad=False, _children=(self, other))

Tensor.__eq__ = eg

def ne(
    self: Tensor,
    other: Any
) -> Tensor:
    """
    Operator of inequality between two tensors

    Args:
        self (Tensor): First input tensor
        other (Any): Second input tensor or other object that can be compared with a tensor

    Returns:
        Tensor: Result of inequality operation
    """

    other = other if isinstance(other, Tensor) else Tensor(other)

    return Tensor(self.data != other.data, requires_grad=False, _children=(self, other))

Tensor.__ne__ = ne    

def lt(
    self: Tensor,
    other: Any
) -> Tensor:
    """
    Operator of less than between two tensors

    Args:
        self (Tensor): First input tensor
        other (Any): Second input tensor or other object that can be compared with a tensor

    Returns:
        Tensor: Result of less than operation
    """

    other = other if isinstance(other, Tensor) else Tensor(other)

    return Tensor(self.data < other.data, requires_grad=False, _children=(self, other))

Tensor.__lt__ = lt

def le(
    self: Tensor,
    other: Any
) -> Tensor:
    """
    Operator of less than or equal to between two tensors

    Args:
        self (Tensor): First input tensor
        other (Any): Second input tensor or other object that can be compared with a tensor

    Returns:
        Tensor: Result of less than or equal to operation
    """

    other = other if isinstance(other, Tensor) else Tensor(other)

    return Tensor(self.data <= other.data, requires_grad=False, _children=(self, other))

Tensor.__le__ = le

def gt(
    self: Tensor,
    other: Any
) -> Tensor:
    """
    Operator of greater than between two tensors

    Args:
        self (Tensor): First input tensor
        other (Any): Second input tensor or other object that can be compared with a tensor

    Returns:
        Tensor: Result of greater than operation
    """

    other = other if isinstance(other, Tensor) else Tensor(other)

    return Tensor(self.data > other.data, requires_grad=False, _children=(self, other))

Tensor.__gt__ = gt

def ge(
    self: Tensor,
    other: Any
) -> Tensor:
    """
    Operator of greater than or equal to between two tensors

    Args:
        self (Tensor): First input tensor
        other (Any): Second input tensor or other object that can be compared with a tensor

    Returns:
        Tensor: Result of greater than or equal to operation
    """

    other = other if isinstance(other, Tensor) else Tensor(other)

    return Tensor(self.data >= other.data, requires_grad=False, _children=(self, other))

Tensor.__ge__ = ge

def add(
    self: Tensor,
    other: Any
) -> Tensor:
    """
    Calculates the sum of the two tensors

    Args:
        self (Tensor): First input tensor
        other (Any): Second input tensor or other object that can be added to a tensor

    Returns:
        Tensor: Tensor with the sum value of two input tensors with the highest compatible shape
    """

    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data + other.data, requires_grad=(self.requires_grad or other.requires_grad), _children=(self, other))

    def _backward():
        # the unbroadcast squeezes the shape of output tensor to the shape of input tensor in case shapes are different after the function
        # so that gradient flows during backward with correct shape
        if self.requires_grad:
            self.grad += unbroadcast(out.grad, self.grad.shape)
        if other.requires_grad:
            other.grad += unbroadcast(out.grad, other.grad.shape)

    out._backward = _backward

    return out

Tensor.__add__ = add

def radd(
    self: Tensor,
    other: Any
) -> Tensor:
    """
    Calculates the sum of the two tensors

    Args:
        self (Tensor): First input tensor
        other (Any): Second input tensor or other object that can be added to a tensor

    Returns:
        Tensor: Tensor with the sum value of two input tensors with the highest compatible shape
    """

    other = other if isinstance(other, Tensor) else Tensor(other)

    return other.__add__(self)

Tensor.__radd__ = radd

def sub(
    self: Tensor,
    other: Any
) -> Tensor:
    """
    Calculates the subtraction of the two tensors

    Args:
        self (Tensor): First input tensor
        other (Any): Second input tensor or other object that can be subtracted from a tensor

    Returns:
        Tensor: Tensor with the subtracted value of two input tensors with the highest compatible shape
    """

    other = other if isinstance(other, Tensor) else Tensor(other)

    return self + (-other)

Tensor.__sub__ = sub

def rsub(
    self: Tensor,
    other: Any
) -> Tensor:
    """
    Calculates the subtraction of the two tensors

    Args:
        self (Tensor): First input tensor
        other (Any): Second input tensor or other object that a tensor can be subtracted from

    Returns:
        Tensor: Tensor with the subtracted value of two input tensors with the highest compatible shape
    """

    other = other if isinstance(other, Tensor) else Tensor(other)

    return other.__sub__(self)

Tensor.__rsub__ = rsub

def neg(self: Tensor) -> Tensor:
    """
    Calculates the negative value of the input tensor

    Args:
        self (Tensor): Input tensor

    Returns:
        Tensor: Tensor with the negative value of the input tensor
    """

    out = Tensor(-self.data, requires_grad=self.requires_grad, _children=(self,))

    def _backward():
        if self.requires_grad:
            self.grad += -out.grad

    out._backward = _backward

    return out

Tensor.__neg__ = neg

def mul(
    self: Tensor,
    other: Any
) -> Tensor:
    """
    Calculates the multiplication of the two tensors

    Args:
        self (Tensor): First input tensor
        other (Any): Second input tensor or other object that can be multiplied to a tensor

    Returns:
        Tensor: Tensor with the multiplied value of two input tensors with the highest compatible shape
    """

    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data * other.data, requires_grad=(self.requires_grad or other.requires_grad), _children=(self, other))

    def _backward():
        if self.requires_grad:
            self.grad += unbroadcast(other.data * out.grad, self.grad.shape)
        if other.requires_grad:
            other.grad += unbroadcast(self.data * out.grad, other.grad.shape)

    out._backward = _backward

    return out

Tensor.__mul__ = mul

def rmul(
    self: Tensor,
    other: Any
) -> Tensor:
    """
    Calculates the multiplication of the two tensors

    Args:
        self (Tensor): First input tensor
        other (Any): Second input tensor or other object that can be multiplied to a tensor

    Returns:
        Tensor: Tensor with the multiplied value of two input tensors with the highest compatible shape
    """

    other = other if isinstance(other, Tensor) else Tensor(other)

    return other.__mul__(self)

Tensor.__rmul__ = rmul

def pow(
    self: Tensor,
    other: Any
) -> Tensor:
    """
    Calculates the power function of the input tensor base any power

    Args:
        self (Tensor): First input tensor
        other (Any): Second input tensor with power value or other object that tensor can be powered with

    Returns:
        Tensor: Tensor with the powered value
    """

    other = other if isinstance(other, Tensor) else Tensor(other)

    out = Tensor(self.data**other.data, requires_grad=(self.requires_grad or other.requires_grad), _children=(self, other))

    def _backward():
        # tabular derivative: d((x)^pow) = pow * x^(pow-1)
        if self.requires_grad:
            self.grad += self.data**(other.data-1) * other.data * out.grad
        # tabular derivative: d(base^x) = base^x * ln(base)
        if other.requires_grad:
            other.grad += self.data**other.data * np.log(self.data) * out.grad
    
    out._backward = _backward

    return out

Tensor.__pow__ = pow

def rpow(
    self: Tensor,
    other: Any
) -> Tensor:
    """
    Calculates the power function of any base and input tensor power

    Args:
        self (Tensor): First input tensor
        other (Any): Second input tensor with base value or other object that can be powered with a tensor

    Returns:
        Tensor: Tensor with the powered value
    """

    other = other if isinstance(other, Tensor) else Tensor(other)

    return other.__pow__(self)

Tensor.__rpow__ = rpow

def truediv(
    self: Tensor,
    other: Any
) -> Tensor:
    """
    Calculates the division of the two tensors

    Args:
        self (Tensor): First input tensor
        other (Any): Second input tensor or other object that a tensor can be divided by

    Returns:
        Tensor: Tensor with the divided value of two input tensors with the highest compatible shape
    """

    return self * other**(-1)

Tensor.__truediv__ = truediv

def rtruediv(
    self: Tensor,
    other: Any
) -> Tensor:
    """
    Calculates the division of the two tensors

    Args:
        self (Tensor): First input tensor
        other (Any): Second input tensor or other object that a tensor can be divided by

    Returns:
        Tensor: Tensor with the divided value of two input tensors with the highest compatible shape
    """

    other = other if isinstance(other, Tensor) else Tensor(other)

    return other.__truediv__(self)

Tensor.__rtruediv__ = rtruediv

def matmul(
    self: Tensor,
    other: Tensor
) -> Tensor:
    """
    Calculates the matrix multiplication of the two tensors

    Args:
        self (Tensor): First input tensor
        other (Tensor): Second input tensor

    Returns:
        Tensor: Tensor with the after the matrix multiplication
    """

    out = Tensor(self.data @ other.data, requires_grad=(self.requires_grad or other.requires_grad), _children=(self, other))

    """
    Matrix gradients:
    dC = dA @ B + A @ dB

    Using dL = tr((dL/dC)^T dC) and trace cyclic property:
    
    ∇A L = (dL/dC) @ B^T
    ∇B L = A^T @ (dL/dC)
    """

    def _backward():
        if self.requires_grad:
            self.grad += out.grad @ other.data.T
        if other.requires_grad:
            other.grad += self.data.T @ out.grad

    out._backward = _backward

    return out

Tensor.__matmul__ = matmul


def sum(
    t: Tensor,
    axis: Tuple[int] = None,
    keepdims: bool = False
) -> Tensor:
    """
    Calculates the sum of the tensor over the given axis

    Args:
        t (Tensor): Input tensor
        axis (None or int or Tuple[int], optional): Axis or axes along which a sum is performed. The default, axis=None, will sum all of the elements of the input tensor. If axis is negative it counts from the last to the first axis.
        keepdims (bool, optional): If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input tensor.

    Returns:
        Tensor: Tensor after the sum function
    """

    out_data = t.data.sum(axis=axis, keepdims=keepdims)
    out = Tensor(out_data, requires_grad=t.requires_grad, _children=(t,))

    # leaves all gradients according to structure
    def _backward():
        if t.requires_grad:
            grad = out.grad
            if not keepdims and axis:
                # returns old dimensions to tensor
                grad = np.expand_dims(grad, axis=axis)
            # broadcast the tensor to its old shape
            t.grad += np.broadcast_to(grad, t.shape)

    out._backward = _backward

    return out


def mean(
    t: Tensor,
    axis: Tuple[int] = None,
    keepdims: bool = False
) -> Tensor:
    """
    Calculates the mean of the tensor over the given axis

    Args:
        t (Tensor): Input tensor
        axis (None or int or Tuple[int], optional): Axis or axes along which the means are computed. The default is to compute the mean of the flattened tensor.
        keepdims (bool, optional): If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array.

    Returns:
        Tensor: Tensor after the mean function
    """

    out_data = np.mean(t.data, axis=axis, keepdims=keepdims)
    out = Tensor(out_data, requires_grad=t.requires_grad, _children=(t,))
    
    # leaves all gradients according to structure
    def _backward():
        if t.requires_grad:
            grad = out.grad
            if not keepdims and axis:
                # returns old dimensions to tensor
                grad = np.expand_dims(grad, axis=axis)
            # broadcast the tensor to its old shape
            if axis:
                if isinstance(axis, Tuple):
                    mean_count = np.sum(list(t.shape[ax] for ax in axis))
                else:
                    mean_count = t.shape[axis]
            else:
                mean_count = np.sum(t.shape)
            t.grad += np.broadcast_to(grad, t.shape) / mean_count

    out._backward = _backward

    return out


def relu(
    t: Tensor
) -> Tensor:
    """
    Equates all negative elements of the input tensor to zero

    Args:
        t (Tensor): Input tensor

    Returns:
        Tensor: Tensor after the RELU function
    """
    out_data = np.maximum(t.data, 0.0)

    out = Tensor(out_data, requires_grad=t.requires_grad, _children=(t,))

    # leaves all gradients where value is positive
    def _backward():
        if t.requires_grad:
            t.grad += (t.data > 0) * out.grad

    out._backward = _backward

    return out


def log(
    t: Tensor
) -> Tensor:
    """
    Calculate the natural logarithm of each element of the input tensor

    Args:
        t (Tensor): Input tensor

    Returns:
        Tensor: Tensor after the logarithm function
    """
    out = Tensor(np.log(t.data), requires_grad=t.requires_grad, _children=(t,))

    # tabular derivative: d(ln(x)) = 1/x
    def _backward():
        if t.requires_grad:
            t.grad += (1 / t.data) * out.grad

    out._backward = _backward

    return out


def exp(
    t: Tensor
) -> Tensor:
    """
    Calculate the exponential function using the base constant e of each element of the input tensor

    Args:
        t (Tensor): Input tensor

    Returns:
        Tensor: Tensor after the exponential function
    """
    out = Tensor(np.exp(t.data), requires_grad=t.requires_grad, _children=(t,))

    # tabular derivative: d(e^x) = e^x
    def _backward():
        if t.requires_grad:
            t.grad += out.data * out.grad
            
    out._backward = _backward

    return out


def clip(
    t: Tensor,
    min: float,
    max: float
) -> Tensor:
    """
    Clip the tensor from min to max

    Args:
        t (Tensor): The input tensor
        min (float): Min number for tensor clipping
        max (float): Max number for tensor clipping

    Returns:
        Tensor: Tensor with the clipped value
    """

    out = Tensor(data=np.clip(t.data, min, max), requires_grad=t.requires_grad, _children=(t._previous))

    def _backward():
        if t.requires_grad:
            t.grad[(t.data >= min) & (t.data <= max)] += out.grad[(t.data >= min) & (t.data <= max)]

    out._backward = _backward

    return out


def contiguous(t: Tensor) -> Tensor:
    """
    Make the tensor have a contiguous area in memory

    Args:
        t (Tensor): Input tensor

    Returns:
        Tensor: contiguous tensor with initial data
    """

    if t.is_contiguous():
        return t
    
    out = Tensor(data=np.ascontiguousarray(t.data), requires_grad=t.requires_grad, _children=t._previous)

    out._backward = t._backward
    
    return out


def view(
    t: Tensor,
    *shape: int
) -> Tensor:
    """
    Method for changing the contiguous input tensor into a different shape

    Args:
        t (Tensor): Input tensor
        shape (Sequence of int): Shape the input tensor needs to be fit in
    
    Returns:
        Tensor: Tensor with changed shape
    """

    if not t.is_contiguous():
        raise RuntimeError("view requires contiguous tensor")

    out = Tensor(t.data.reshape(*shape, copy=False), requires_grad=t.requires_grad, _children=(t,))

    def _backward():
        # during backward we need to reshape the gradient of the output tensor into the shape of the input tensor
        if t.requires_grad:
            t.grad += out.grad.reshape(t.shape)

    out._backward = _backward

    return out


def view_as(
    t1: Tensor,
    t2: Tensor
) -> Tensor:
    """
    Method for changing the first contiguous input tensor into a shape of the second input tensor

    Args:
        t1 (Tensor): First input tensor
        t2 (Tensor): Second input tensor
    
    Returns:
        Tensor: Tensor with changed shape
    """

    return view(t1, t2.shape)


def reshape(
    t: Tensor,
    *shape: int
) -> Tensor:
    """
    Method for changing the input tensor into a different shape
    If tensor is contiguous, equal to view

    Args:
        t (Tensor): Input tensor
        shape (Sequence of int): Shape the input tensor needs to be fit in
    
    Returns:
        Tensor: Tensor with changed shape
    """

    try:
        return view(t, *shape)
    except RuntimeError:
        return view(contiguous(t), *shape)
    

def reshape_as(
    t1: Tensor,
    t2: Tensor
) -> Tensor:
    """
    Method for changing the first input tensor into a shape of the second input tensor
    If tensor is contiguous, equal to view

    Args:
        t1 (Tensor): First input tensor
        t2 (Tensor): Second input tensor
    
    Returns:
        Tensor: Tensor with changed shape
    """

    return reshape(t1, t2.shape)


def t(t: Tensor) -> Tensor:
    """
    Method for transposing the input tensor

    Args:
        t (Tensor): Input tensor

    Returns:
        Tensor: Transposed tensor
    """

    if t.ndim <= 1:
        return t

    if t.ndim != 2:
        raise RuntimeError(
            "t() expects a tensor with <= 2 dimensions"
        )

    return transpose(t, 0, 1)


def transpose(
    t: Tensor,
    dim0: int,
    dim1: int
) -> Tensor:
    """
    Method for swapping two of axes of the input tensor

    Args:
        t (Tensor): Input tensor
        dim0 (int): First dimension to swap
        dim1 (int): Second dimension to swap

    Returns:
        Tensor: Tensor with two axes swapped
    """

    axes = list(range(t.ndim))
    axes[dim0], axes[dim1] = axes[dim1], axes[dim0]

    return permute(t, *axes)


def swapaxes(
    t: Tensor,
    dim0: int,
    dim1: int
) -> Tensor:
    """
    Method for swapping two of axes of the input tensor

    Args:
        t (Tensor): Input tensor
        dim0 (int): First dimension to swap
        dim1 (int): Second dimension to swap

    Returns:
        Tensor: Tensor with two axes swapped
    """

    return transpose(t, dim0, dim1)


def movedim(
    t: Tensor,
    source: Tuple[int],
    destination: Tuple[int]
) -> Tensor:
    """
    Method for moving axes of the input tensor to new position

    Args:
        t (Tensor): Input tensor
        source (Tuple[int]): Indices of dimensions to move
        destination (Tuple[int]): Indices for insertion of dimensions

    Returns:
        Tensor: Tensor with moved dimensions
    """

    ndim = t.ndim

    if isinstance(source, int):
        source = (source,)

    if isinstance(destination, int):
        destination = (destination,)

    if len(source) != len(destination):
        raise ValueError("source and destination must have the same number of dimensions")
    
    source_norm = []
    destination_norm = []

    for s, d in zip(source, destination):
        if not (-ndim <= s < ndim) or not (-ndim <= d < ndim):
            raise IndexError("index out of range")
        
        source_norm.append(s % ndim)
        destination_norm.append(d % ndim)

    if len(source_norm) != len(set(source_norm)):
        raise ValueError("source must not have the repeated indices")
    
    if len(destination_norm) != len(set(destination_norm)):
        raise ValueError("destination must not have the repeated indices")

    remaining = [i for i in range(ndim) if i not in source]

    permutation = [None] * ndim

    for s, d in zip(source_norm, destination_norm):
        permutation[d] = s

    remaining_it = iter(remaining)

    for i in range(ndim):
        if permutation[i] is None:
            permutation[i] = next(remaining_it)

    return permute(t, *permutation)


def moveaxis(
    t: Tensor,
    source: Tuple[int],
    destination: Tuple[int]
) -> Tensor:
    """
    Method for moving axes of the input tensor to new position

    Args:
        t (Tensor): Input tensor
        source (Tuple[int]): Indices of dimensions to move
        destination (Tuple[int]): Indices for insertion of dimensions

    Returns:
        Tensor: Tensor with moved dimensions
    """

    return movedim(t, source, destination)


def permute(
    t: Tensor,
    *dims: int
) -> Tensor:
    """
    Method for changing order of dimensions of the input tensor

    Args:
        t (Tensor): Input tensor
        dims (Sequence of int): Order of the dimensions of the input tensor the output tensor needs to be with

    Returns:
        Tensor: Tensor with changed order of dimensions
    """

    out = Tensor(np.transpose(t.data, dims), requires_grad=t.requires_grad, _children=(t,))

    def _backward():
        # during backward we need to transpose the gradient of the output tensor into the dims order of the input tensor
        if t.requires_grad:
            reversed_dims = np.argsort(dims)
            t.grad += np.transpose(out.grad, reversed_dims)

    out._backward = _backward

    return out


def flatten(t: Tensor) -> Tensor:
    """
    Method for flattening a tensor into a vector

    Args:
        t (Tensor): Input tensor

    Returns:
        Tensor: Output one-dimensional tensor
    """

    out = Tensor(t.data.flatten(), requires_grad=t.requires_grad, _children=(t,))

    def _backward():
        if t.requires_grad:
            t.grad += out.grad.reshape(t.shape)

    out._backward = _backward

    return out


def unflatten(
    t: Tensor,
    dim: int,
    *sizes: int
) -> Tensor:
    """
    Method for changing a tensor dimension into a new shape

    Args:
        t (Tensor): Input tensor
        dim (int): Dimension to unflatten
        sizes (Sequence of int): Shape the dimension needs to be fit in

    Returns:
        Tensor: Output tensor with new shape
    """

    ndim = t.ndim
    shape = t.shape

    if not (-ndim <= dim < ndim):
        raise IndexError("index out of range")
    
    dim %= t.ndim

    new_shape = shape[:dim] + sizes + shape[dim + 1:]

    return reshape(t, *new_shape)


def expand(
    t: Tensor,
    *shape: int
) -> Tensor:
    """
    Method for changing the input tensor to a larger shape

    Args:
        t (Tensor): Input tensor
        shape (Sequence of int): Shape the input tensor needs to be expanded to

    Returns:
        Tensor: Tensor with expanded shape
    """

    out = Tensor(np.broadcast_to(t.data, shape), requires_grad=t.requires_grad, _children=(t,))

    def _backward():
        # during backward we need to change back the gradient shape into the smaller shape of the input tensor
        if t.requires_grad:
            t.grad += unbroadcast(out.grad, t.shape)

    out._backward = _backward

    return out


def expand_as(
    t1: Tensor,
    t2: Tensor
) -> Tensor:
    """
    Method for changing the first input tensor into a larger shape of the second input tensor

    Args:
        t1 (Tensor): First input tensor
        t2 (Tensor): Second input tensor
    
    Returns:
        Tensor: Tensor with expanded shape
    """

    return expand(t1, t2.shape)


def broadcast_to(
    t: Tensor,
    *shape: int
) -> Tensor:
    """
    Method for changing the input tensor to a larger shape

    Args:
        t (Tensor): Input tensor
        shape (Sequence of int): Shape the input tensor needs to be broadcasted to

    Returns:
        Tensor: Tensor with broadcasted shape
    """

    return expand(t, shape)


def squeeze(
    t: Tensor,
    dim: Tuple[int] = None
) -> Tensor:
    """
    Method for removing dimensions of length 1 from the input tensor

    Args:
        t (Tensor): Input tensor
        dim (None or int or Tuple[int]): Dimensions with lenght 1 which need to be removed from the input tensor

    Returns:
        Tensor: Tensor with removed dimensions
    """

    out = Tensor(np.squeeze(t.data, axis=dim), requires_grad=t.requires_grad, _children=(t,))

    def _backward():
        if t.requires_grad:
            t.grad += out.grad.reshape(t.shape)

    out._backward = _backward

    return out


def unsqueeze(
    t: Tensor,
    dim: Tuple[int]
) -> Tensor:
    """
    Method for adding new dimensions to the input tensor

    Args:
        t (Tensor): Input tensor
        dim (int or Tuple[int]): Positions where the new dimensions will be added

    Returns:
        Tensor: Tensor with added dimensions
    """

    out = Tensor(np.expand_dims(t.data, axis=dim), requires_grad=t.requires_grad, _children=(t,))

    def _backward():
        if t.requires_grad:
            t.grad += out.grad.reshape(t.shape)

    out._backward = _backward
    
    return out


def split(
    t: Tensor,
    split_size_or_sections: List[int],
    dim: int = 0
) -> Tuple[Tensor]:
    """
    Method for splitting the tensor into multiple ones along one dimension

    Args:
        t (Tensor): Input tensor
        split_size_or_sections (int or List[int]): Size of every chunk for splitting or range of sizes for all chunks
        dim (int): Dimension the tensor is being splitted along

    Returns:
        Tuple[Tensor]: Tuple with views of initial tensor
    """

    ndim = t.ndim

    if not (-ndim <= dim < ndim):
        raise IndexError("index out of range")
    
    dim %= t.ndim

    chunks = []

    if isinstance(split_size_or_sections, int):
        if split_size_or_sections <= 0:
            raise ValueError("split size must be positive")
        
        start = 0

        while start < t.shape[dim]:
            stop = min(start + split_size_or_sections, t.shape[dim])

            slices = [slice(None)] * t.ndim
            slices[dim] = slice(start, stop)

            tensor = Tensor(t.data[tuple(slices)], requires_grad=t.requires_grad, _children=(t,))

            def _backward(tensor=tensor, slices=tuple(slices)):
                if t.requires_grad:
                    t.grad[slices] += tensor.grad

            tensor._backward = _backward

            chunks.append(tensor)

            start = stop
    else:
        if any(size <= 0 for size in split_size_or_sections):
            raise ValueError("split size must be positive")
        if np.sum(split_size_or_sections) != t.shape[dim]:
            raise ValueError("sum of sections must match the size of dimension")

        start = 0
        stop = 0

        for size in split_size_or_sections:
            stop += size
            
            slices = [slice(None)] * t.ndim
            slices[dim] = slice(start, stop)

            tensor = Tensor(t.data[tuple(slices)], requires_grad=t.requires_grad, _children=(t,))

            def _backward(tensor=tensor, slices=tuple(slices)):
                if t.requires_grad:
                    t.grad[slices] += tensor.grad

            tensor._backward = _backward

            chunks.append(tensor)

            start = stop
    
    return tuple(chunks)


def chunk(
    t: Tensor,
    chunks: int,
    dim: int = 0
) -> Tuple[Tensor]:
    """
    Method for splitting the tensor into multiple ones along one dimension

    Args:
        t (Tensor): Input tensor
        chunks (int): Number of chunks for splitting
        dim (int): Dimension the tensor is being splitted along

    Returns:
        Tuple[Tensor]: Tuple with views of initial tensor
    """

    if chunks <= 0:
        raise ValueError("number of chunks must be positive")
    
    chunk_size = int(np.ceil(t.shape[dim] / chunks))

    return split(t, chunk_size, dim)


def atleast_1d(*tensors: Tensor) -> Tuple[Tensor]:
    """
    Method for reshaping tensor into at least one-dimensional tensor. If tensor has 1 or more dimensions, doesn't change an initial tensor

    Args:
        tensors (Tensor or Sequence of Tensor): Input tensors

    Returns:
        (Tuple[Tensor]): Tensors with at least one dimension
    """

    atleast_1d_tensors = []

    for tensor in tensors:
        if tensor.ndim < 1:
            tensor = unsqueeze(tensor, dim=0)

        atleast_1d_tensors.append(tensor)

    return tuple(atleast_1d_tensors)


def atleast_2d(*tensors: Tensor) -> Tuple[Tensor]:
    """
    Method for reshaping tensor into at least two-dimensional tensor. If tensor has 2 or more dimensions, doesn't change an initial tensor

    Args:
        tensors (Tensor or Sequence of Tensor): Input tensors

    Returns:
        (Tuple[Tensor]): Tensors with at least two dimensions
    """

    atleast_2d_tensors = []

    tensors = atleast_1d(*tensors)

    for tensor in tensors:
        if tensor.ndim == 1:
            tensor = unsqueeze(tensor, dim=0)

        atleast_2d_tensors.append(tensor)

    return tuple(atleast_2d_tensors)


def atleast_3d(*tensors: Tensor) -> Tuple[Tensor]:
    """
    Method for reshaping tensor into at least three-dimensional tensor. If tensor has 3 or more dimensions, doesn't change an initial tensor

    Args:
        tensors (Tensor or Sequence of Tensor): Input tensors

    Returns:
        (Tuple[Tensor]): Tensors with at least three dimensions
    """

    atleast_3d_tensors = []

    tensors = atleast_2d(*tensors)

    for tensor in tensors:
        if tensor.ndim == 2:
            tensor = unsqueeze(tensor, dim=-1)

        atleast_3d_tensors.append(tensor)

    return tuple(atleast_3d_tensors)


def cat(
    tensors: Tuple[Tensor],
    dim: int = 0
) -> Tensor:
    """
    Method for concatenating tuple of tensors into one tensor along one dimension

    Args:
        tensors (Tuple[Tensor]): Tuple of input tensors
        dim (int): Dimension the tensors are being concatenated along

    Returns:
        Tensor: Tensor with view of concatenated initial tensors
    """

    out = Tensor(
        data=np.concatenate([tensor.data for tensor in tensors], axis=dim),
        requires_grad=bool(np.any([tensor.requires_grad for tensor in tensors])),
        _children=tensors
    )

    def _backward():
        start = 0
        stop = 0

        for tensor in tensors:
            stop += tensor.shape[dim]

            slices = [slice(None)] * tensor.ndim
            slices[dim] = slice(start, stop)

            if tensor.requires_grad:
                tensor.grad += out.grad[tuple(slices)]

            start = stop

    out._backward = _backward

    return out


def stack(
    tensors: Tuple[Tensor],
    dim: int = 0
) -> Tensor:
    """
    Method for stacking tuple of tensors into one tensor along a new dimension

    Args:
        tensors (Tuple[Tensor]): Tuple of input tensors
        dim (int): New dimension the tensors are being stacked along

    Returns:
        Tensor: Tensor with view of stacked initial tensors
    """

    unsqueezed_tensors = tuple(unsqueeze(tensor, dim=dim) for tensor in tensors)

    return cat(unsqueezed_tensors, dim)


def vstack(tensors: Tuple[Tensor]) -> Tensor:
    """
    Method for stacking tuple of tensors into one tensor along vertical dimension

    Args:
        tensors (Tuple[Tensor]): Tuple of input tensors

    Returns:
        Tensor: Tensor with view of stacked initial tensors
    """
    
    # we need to stack along the vertical dimension H, so the tensors must have a shape at least (H, W)
    tensors = atleast_2d(*tensors)

    return cat(tuple(tensors), dim=0)


def hstack(tensors: Tuple[Tensor]) -> Tensor:
    """
    Method for stacking tuple of tensors into one tensor along horizontal dimension

    Args:
        tensors (Tuple[Tensor]): Tuple of input tensors

    Returns:
        Tensor: Tensor with view of stacked initial tensors
    """

    # we need to stack along the horisontal dimension W, so a shape (W,) is enough for operation
    tensors = atleast_1d(*tensors)

    if tensors[0].ndim == 1:
        return cat(tensors, dim=0)
    else:
        return cat(tensors, dim=1)
    

def dstack(tensors: Tuple[Tensor]) -> Tensor:
    """
    Method for stacking tuple of tensors into one tensor along depth dimension

    Args:
        tensors (Tuple[Tensor]): Tuple of input tensors

    Returns:
        Tensor: Tensor with view of stacked initial tensors
    """

    # we need to stack along the depth dimension D, so the tensors must have a shape at least (H, W, D)
    tensors = atleast_3d(*tensors)

    return cat(tuple(tensors), dim=2)


def repeat(
    t: Tensor,
    repeats: Tuple[int]
) -> Tensor:
    """
    Method for constructing a tensor by repeating specified dimensions of elemenst

    Args:
        t (Tensor): Input tensor
        repeats (Tuple[int]): Number of each dimension to be repeated

    Returns:
        Tensor: Tensor with repeated elements
    """

    out = Tensor(data=np.tile(t.data, reps=repeats), requires_grad=t.requires_grad, _children=(t,))

    def _backward(repeats=repeats):
        if t.requires_grad:
            shape = t.shape

            if len(repeats) >= t.ndim:
                shape = (1,) * (len(repeats) - t.ndim) + t.shape  
            else:
                repeats = (1,) * (t.ndim - len(repeats)) + repeats

            reshape = []

            for d, r in zip(shape, repeats):
                reshape.extend([r, d])

            grad = np.reshape(out.grad, shape=reshape)

            reduce_dims = tuple(range(0, len(reshape), 2))

            t.grad += np.sum(grad, axis=reduce_dims, keepdims=False)

    out._backward = _backward

    return out


def tile(
    t: Tensor,
    dims: Tuple[int]
) -> Tensor:
    """
    Method for constructing a tensor by repeating specified dimensions of elemenst

    Args:
        t (Tensor): Input tensor
        dims (Tuple[int]): Number of times tensor values to be repeated along each dimension

    Returns:
        Tensor: Tensor with repeated elements
    """

    return repeat(t, dims)


def repeat_interleave(
    t: Tensor,
    repeats: Tuple[int],
    dim: int | None = None
) -> Tensor:
    """
    Method for repeating elements of the input tensor

    Args:
        t (Tensor): Input tensor
        repeats (int or Tuple[int]): The number of repetition for each element
        dim (int, optional): The dimension along which to repeat the elements

    Returns:
        Tensor: Tensor with repeated elements
    """

    out = Tensor(data=np.repeat(t.data, repeats, axis=dim), requires_grad=t.requires_grad, _children=(t,))

    if not dim:
        dim = 0
    else:
        dim %= t.ndim

    def _backward():
        if t.requires_grad:
            if isinstance(repeats, int):
                reshape = []

                for d in range(t.ndim):
                    reshape.append(t.shape[d])

                    if d == dim:
                        reshape.append(repeats)

                grad = np.reshape(out.grad, shape=reshape)

                t.grad += np.sum(grad, axis=dim+1, keepdims=False)
            else:
                start = 0
                stop = 0

                for repeat in repeats:
                    stop += repeat
                    slices = [slice(None)] * t.ndim
                    slices[dim] = slice(start, stop)

                    t.grad[tuple(slices)] += np.sum(out.grad[tuple(slices)], axis=dim, keepdims=True)

                    start = stop

    out._backward = _backward

    return out


def clone(
    t: Tensor
) -> Tensor:
    """
    Method for cloning a tensor

    Args:
        t (Tensor): Input tensor to clone

    Returns:
        Tensor: Cloned tensor
    """

    out = Tensor(data=t.data, requires_grad=t.requires_grad, _children=(t,), copy=True)

    def _backward():
        if t.requires_grad:
            t.grad += out.grad

    out._backward = _backward

    return out


def type(
    t: Tensor,
    dtype: np.dtype | None = None
) -> np.dtype | Tensor:
    """
    Method for casting a tensor into necessary type

    Args:
        t (Tensor): Input tensor
        dtype(np.dtype, optional): Type for casting tensor

    Returns:
        Tensor: Tensor with necessary type
    """
    
    if dtype:
        if dtype in (np.float16, np.float32, np.float64):
            out = Tensor(
                data=t.data.astype(dtype),
                requires_grad=t.requires_grad,
                _children=(t,),
                dtype=dtype
            )

            def _backward():
                if t.requires_grad:
                    t.grad += out.grad.astype(t.dtype)

            out._backward = _backward
        else:
            out = Tensor(
                data=t.data.astype(dtype),
                requires_grad=False,
                _children=(t,),
                dtype=dtype
            )

        return out
    else:
        return t.dtype


def gather(
    t: Tensor,
    dim: int,
    index: Tensor
) -> Tensor:
    """
    Method for gathering values from input tensor along the particular dimension using indices from index tensor

    Args:
        t (Tensor): Input tensor
        dim (int): Dimension along which values are gathered
        index (Tensor): Tensor with indices for gathering

    Returns:
        Tensor: Tensor with values gathered along the dimension according to indices
    """

    if not isinstance(index, Tensor):
        index = Tensor(index, dtype=np.int32)

    if t.ndim != index.ndim:
        raise ValueError("input tensor and index tensor must have the same number of dimensions")

    if index.dtype not in (np.int8, np.int16, np.int32, np.int64):
        index = type(index, np.int32)
    
    out = Tensor(
        data=np.take_along_axis(t.data, index.data, axis=dim),
        requires_grad=t.requires_grad,
        _children=(t,)
    )

    def _backward():
        if t.requires_grad:
            coords = list(np.indices(index.shape, sparse=True))
            coords[dim] = index.data
            flat_indices = np.ravel_multi_index(
                tuple(coords),
                t.shape,
            )

            np.add.at(
                t.grad.ravel(),
                flat_indices.ravel(),
                out.grad.ravel(),
            )

    out._backward = _backward

    return out


def scatter(
    target: Tensor,
    dim: int,
    src: Tensor,
    index: Tensor
) -> Tensor:
    """
    Method for putting values from source tensor to target tensor along the particular dimension using indices from index tensor

    Args:
        target (Tensor): Input tensor to which put values
        dim (int): Dimension along which values are put
        src (Tensor): Tensor with values to put
        index (Tensor): Tensor with indices for putting

    Returns:
        Tensor: Tensor with values put along the dimension according to indices
    """

    if not isinstance(index, Tensor):
        index = Tensor(index, dtype=np.int32)
    
    if target.ndim != index.ndim:
        raise ValueError("target tensor and index tensor must have the same number of dimensions")

    if src.ndim != index.ndim:
        raise ValueError("source tensor and index tensor must have the same number of dimensions")

    if index.dtype not in (np.int8, np.int16, np.int32, np.int64):
        index = type(index, np.int32)

    if has_duplicates_along_axis(array=index.data, axis=dim):
        raise ValueError("index cannot contain identical indices along target dimension")

    target_clone = clone(target)

    np.put_along_axis(target_clone.data, index.data, src.data, axis=dim)

    out = Tensor(
        data=target_clone.data,
        requires_grad=(target_clone.requires_grad or src.requires_grad),
        _children=(target_clone, src)
    )

    def _backward():
        if target_clone.requires_grad:
            overwritten = np.zeros(out.grad.shape, dtype=bool)
            np.put_along_axis(overwritten, index.data, True, axis=dim)
            not_overwritten = ~overwritten

            target_clone.grad[not_overwritten] += out.grad[not_overwritten]
        if src.requires_grad:
            src.grad += np.take_along_axis(out.grad, index.data, axis=dim)

    out._backward = _backward

    return out


def scatter_add(
    target: Tensor,
    dim: int,
    src: Tensor,
    index: Tensor
) -> Tensor:
    """
    Method for adding values from source tensor to target tensor along the particular dimension using indices from index tensor

    Args:
        target (Tensor): Input tensor to which add values
        dim (int): Dimension along which values are added
        src (Tensor): Tensor with values to add
        index (Tensor): Tensor with indices for adding

    Returns:
        Tensor: Tensor with values summed along the dimension according to indices
    """

    if not isinstance(index, Tensor):
        index = Tensor(index, dtype=np.int32)

    if target.ndim != index.ndim:
        raise ValueError("target tensor and index tensor must have the same number of dimensions")

    if src.ndim != index.ndim:
        raise ValueError("source tensor and index tensor must have the same number of dimensions")

    if index.dtype not in (np.int8, np.int16, np.int32, np.int64):
        index = type(index, np.int32)

    coords = list(np.indices(index.shape, sparse=True))
    coords[dim] = index.data
    flat_indices = np.ravel_multi_index(
        tuple(coords),
        target.shape,
    )

    target_clone = clone(target)

    np.add.at(
        target_clone.data.ravel(),
        flat_indices.ravel(),
        src.data.ravel(),
    )

    out = Tensor(
        data=target_clone.data,
        requires_grad=(target_clone.requires_grad or src.requires_grad),
        _children=(target_clone, src)
    )

    def _backward():
        if target_clone.requires_grad:
            target_clone.grad += out.grad

        if src.requires_grad:
            src.grad += np.take_along_axis(out.grad, index.data, axis=dim)

    out._backward = _backward

    return out


def index_select(
    t: Tensor,
    dim: int,
    index: Tensor
) -> Tensor:
    """
    Method for taking values from input tensor along the particular dimension using indices from index tensor

    Args:
        t (Tensor): Input tensor
        dim (int): Dimension along which values are taken
        index (Tensor): Tensor with indices for taking values

    Returns:
        Tensor: Tensor with values taken along the dimension according to indices
    """

    if not isinstance(index, Tensor):
        index = Tensor(index, dtype=np.int32)

    if index.dtype not in (np.int8, np.int16, np.int32, np.int64):
        index = type(index, np.int32)

    if index.ndim > 1:
        raise ValueError("index tensor must have 1 or less dimensions")

    if not (-t.ndim <= dim < t.ndim):
        raise IndexError("index out of range")
        
    dim %= t.ndim

    indices = tuple(index.data if i == dim else slice(None) for i in range(t.ndim))

    out = Tensor(data=t.data[indices], requires_grad=t.requires_grad, _children=(t,))

    def _backward():
        if t.requires_grad:
            if index.ndim > 0:
                index_coefficients = element_counts_along_axis(index.data, axis=0)
            else:
                index_coefficients = 1

            t.grad[indices] += out.grad * index_coefficients

    out._backward = _backward

    return out


def take(
    t: Tensor,
    index: Tensor
):
    """
    Method for taking values from input tensor viewed as a 1-D tensor using indices from index tensor
    
    Args:
        t (Tensor): Input tensor
        index (Tensor): Tensor with indices for taking values

    Returns:
        Tensor: Tensor with values taken according to indices
    """

    if not isinstance(index, Tensor):
        index = Tensor(index, dtype=np.int32)

    if index.dtype not in (np.int8, np.int16, np.int32, np.int64):
        index = type(index, np.int32)

    index_raveled = index.data.ravel()

    flattened_data = t.data.ravel()[index_raveled]

    out = Tensor(
        data=np.reshape(flattened_data, index.shape),
        requires_grad=t.requires_grad,
        _children=(t,)
    )

    def _backward():
        if t.requires_grad:
            if index.ndim > 0:
                index_coefficients = element_counts_along_axis(index_raveled, axis=0)
            else:
                index_coefficients = 1

            grad_flattened = t.grad.ravel()
            grad_flattened[index_raveled] += out.grad.ravel() * index_coefficients

            t.grad = np.reshape(grad_flattened, t.shape)

    out._backward = _backward

    return out


def masked_select (
    t: Tensor,
    mask: Tensor
) -> Tensor:
    """
    Method for taking values from input tensor using boolean mask tensor
    
    Args:
        t (Tensor): Input tensor
        mask (Tensor): Tensor with bool values for taking values

    Returns:
        Tensor: Tensor with values taken according to mask
    """

    if not isinstance(mask, Tensor):
        mask = Tensor(mask, dtype=np.bool)

    if mask.dtype != bool:
        mask = type(mask, bool)

    if mask.shape != t.shape:
        mask_raveled = np.broadcast_to(mask.data, t.shape).ravel()
    else:
        mask_raveled = mask.data.ravel()

    out = Tensor(data=t.data.ravel()[mask_raveled], requires_grad=t.requires_grad, _children=(t,))

    def _backward():
        if t.requires_grad:
            grad_flattened = t.grad.ravel()
            grad_flattened[mask_raveled] += out.grad.ravel()

            t.grad = np.reshape(grad_flattened, t.shape)

    out._backward = _backward

    return out


def nonzero(
    t: Tensor,
    as_tuple: bool = False
) -> Tuple[Tensor] | Tensor:
    """
    Method for finding indices of nonzero values in input tensor

    Args:
        t (Tensor): Input tensor
        as_tuple: If False the output tensor contains indices. If True the output is tuple with one 1-D tensor for each dimension, containing the indices of each nonzero element along that dimension.

    
    Returns:
        Tensor or Tuple[Tensor]: Output tensor or tuple of tensors with indices of nonzero values
    """

    if as_tuple:
        out = tuple(
            Tensor(data=data, requires_grad=False, _children=(t,), dtype=np.int32)
            for data in np.nonzero(t.data)
        )
    else:
        out = Tensor(
            data=np.stack(np.nonzero(t.data), axis=1),
            requires_grad=False,
            _children=(t,),
            dtype=np.int32
        )

    return out
