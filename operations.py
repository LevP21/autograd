from typing import Tuple

import numpy as np

from tensor import Tensor
from utils import unbroadcast

def eg(self: Tensor, other: Tensor) -> Tensor:
    """
    Operator of equality between two tensors

    Args:
        self (Tensor): First input tensor
        other (Tensor): Second input tensor

    Returns:
        bool: Result of equality operation
    """

    return self.data == other.data

Tensor.__eq__ = eg

def ne(self: Tensor, other: Tensor) -> Tensor:
    """
    Operator of inequality between two tensors

    Args:
        self (Tensor): First input tensor
        other (Tensor): Second input tensor

    Returns:
        bool: Result of inequality operation
    """

    return self.data != other.data

Tensor.__ne__ = ne    

def lt(self: Tensor, other: Tensor) -> Tensor:
    """
    Operator of less than between two tensors

    Args:
        self (Tensor): First input tensor
        other (Tensor): Second input tensor

    Returns:
        bool: Result of less than operation
    """

    return self.data < other.data

Tensor.__lt__ = lt

def le(self: Tensor, other: Tensor) -> Tensor:
    """
    Operator of less than or equal to between two tensors

    Args:
        self (Tensor): First input tensor
        other (Tensor): Second input tensor

    Returns:
        bool: Result of less than or equal to operation
    """

    return self.data <= other.data

Tensor.__le__ = le

def gt(self: Tensor, other: Tensor) -> Tensor:
    """
    Operator of greater than between two tensors

    Args:
        self (Tensor): First input tensor
        other (Tensor): Second input tensor

    Returns:
        bool: Result of greater than operation
    """

    return self.data > other.data

Tensor.__gt__ = gt

def ge(self: Tensor, other: Tensor) -> Tensor:
    """
    Operator of greater than or equal to between two tensors

    Args:
        self (Tensor): First input tensor
        other (Tensor): Second input tensor

    Returns:
        bool: Result of greater than or equal to operation
    """

    return self.data >= other.data

Tensor.__ge__ = ge

def add(self: Tensor, other) -> Tensor:
    """
    Calculates the sum of the two tensors

    Args:
        self (Tensor): First input tensor
        other (Any): Second input tensor or other object that can be added to a tensor

    Returns:
        Tensor: Tensor with the sum value of two input tensors with the highest compatible shape
    """

    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other))

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

def radd(self: Tensor, other) -> Tensor:
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

def sub(self: Tensor, other) -> Tensor:
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

def rsub(self: Tensor, other) -> Tensor:
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

def mul(self: Tensor, other) -> Tensor:
    """
    Calculates the multiplication of the two tensors

    Args:
        self (Tensor): First input tensor
        other (Any): Second input tensor or other object that can be multiplied to a tensor

    Returns:
        Tensor: Tensor with the multiplied value of two input tensors with the highest compatible shape
    """

    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other))

    def _backward():
        if self.requires_grad:
            self.grad += unbroadcast(other.data * out.grad, self.grad.shape)
        if other.requires_grad:
            other.grad += unbroadcast(self.data * out.grad, other.grad.shape)

    out._backward = _backward

    return out

Tensor.__mul__ = mul

def rmul(self: Tensor, other) -> Tensor:
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

def pow(self: Tensor, other) -> Tensor:
    """
    Calculates the power function of the input tensor base any power

    Args:
        self (Tensor): First input tensor
        other (Any): Second input tensor with power value or other object that tensor can be powered with

    Returns:
        Tensor: Tensor with the powered value
    """

    other = other if isinstance(other, Tensor) else Tensor(other)

    out = Tensor(self.data**other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other))

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

def rpow(self: Tensor, other) -> Tensor:
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

def truediv(self: Tensor, other) -> Tensor:
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

def rtruediv(self: Tensor, other) -> Tensor:
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

def matmul(self: Tensor, other: Tensor) -> Tensor:
    """
    Calculates the matrix multiplication of the two tensors

    Args:
        self (Tensor): First input tensor
        other (Tensor): Second input tensor

    Returns:
        Tensor: Tensor with the after the matrix multiplication
    """

    out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other))

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


class Operator():
    """
    Class for operations with Tensors
    """

    def sum(self, t: Tensor, axis=None, keepdims=False) -> Tensor:
        """
        Calculates the sum of the tensor over the given axis

        Args:
            t (Tensor): Input tensor
            axis (None or int or tuple of ints, optional): Axis or axes along which a sum is performed. The default, axis=None, will sum all of the elements of the input tensor. If axis is negative it counts from the last to the first axis.
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


    def mean(self, t: Tensor, axis=None, keepdims=False):
        """
        Calculates the mean of the tensor over the given axis

        Args:
            t (Tensor): Input tensor
            axis (None or int or tuple of ints, optional): Axis or axes along which the means are computed. The default is to compute the mean of the flattened tensor.
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
                        mean_count = sum((t.shape[ax] for ax in axis))
                    else:
                        mean_count = t.shape[axis]
                else:
                    mean_count = sum(t.shape)
                t.grad += np.broadcast_to(grad, t.shape) / mean_count

        out._backward = _backward

        return out


    def relu(self, t: Tensor) -> Tensor:
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


    def log(self, t: Tensor) -> Tensor:
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


    def exp(self, t: Tensor) -> Tensor:
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
    

    def clip(self, t: Tensor, min: float, max: float):
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
    

    def contiguous(self, t: Tensor):
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
    

    def view(self, t: Tensor, *shape):
        """
        Method for changing the contiguous input tensor into a different shape

        Args:
            t (Tensor): Input tensor
            shape (Tuple[int]): Shape the input tensor needs to be fit in
        
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
    

    def view_as(self, t1: Tensor, t2: Tensor):
        """
        Method for changing the first contiguous input tensor into a shape of the second input tensor

        Args:
            t1 (Tensor): First input tensor
            t2 (Tensor): Second input tensor
        
        Returns:
            Tensor: Tensor with changed shape
        """

        return self.view(t1, t2.shape)
    

    def reshape(self, t: Tensor, *shape):
        """
        Method for changing the input tensor into a different shape
        If tensor is contiguous, equal to view

        Args:
            t (Tensor): Input tensor
            shape (Tuple[int]): Shape the input tensor needs to be fit in
        
        Returns:
            Tensor: Tensor with changed shape
        """

        try:
            return self.view(t, *shape)
        except RuntimeError:
            return self.view(self.contiguous(t), *shape)
        

    def reshape_as(self, t1: Tensor, t2: Tensor):
        """
        Method for changing the first input tensor into a shape of the second input tensor
        If tensor is contiguous, equal to view

        Args:
            t1 (Tensor): First input tensor
            t2 (Tensor): Second input tensor
        
        Returns:
            Tensor: Tensor with changed shape
        """

        return self.reshape(t1, t2.shape)
    

    def t(self, t: Tensor):
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

        return self.transpose(t, 0, 1)
    

    def transpose(self, t: Tensor, dim0: int, dim1: int):
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

        return self.permute(t, *axes)
    

    def swapaxes(self, t: Tensor, dim0: int, dim1: int):
        """
        Method for swapping two of axes of the input tensor

        Args:
            t (Tensor): Input tensor
            dim0 (int): First dimension to swap
            dim1 (int): Second dimension to swap

        Returns:
            Tensor: Tensor with two axes swapped
        """

        return self.transpose(t, dim0, dim1)
    

    def movedim(self, t: Tensor, source, destination):
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
            raise ValueError(
                "source and destination must have the same number of dimensions"
            )
        
        source_norm = []
        destination_norm = []

        for s, d in zip(source, destination):
            if not (-ndim <= s < ndim) or not (-ndim <= d < ndim):
                raise IndexError("index out of range")
            
            source_norm.append(s % ndim)
            destination_norm.append(d % ndim)

        if len(source_norm) != len(set(source_norm)):
            raise ValueError(
                "source must not have the repeated indices"
            )
        
        if len(destination_norm) != len(set(destination_norm)):
            raise ValueError(
                "destination must not have the repeated indices"
            )

        remaining = [i for i in range(ndim) if i not in source]

        permutation = [None] * ndim

        for s, d in zip(source_norm, destination_norm):
            permutation[d] = s

        remaining_it = iter(remaining)

        for i in range(ndim):
            if permutation[i] is None:
                permutation[i] = next(remaining_it)

        return self.permute(t, *permutation)
    

    def moveaxis(self, t: Tensor, source, destination):
        """
        Method for moving axes of the input tensor to new position

        Args:
            t (Tensor): Input tensor
            source (Tuple[int]): Indices of dimensions to move
            destination (Tuple[int]): Indices for insertion of dimensions

        Returns:
            Tensor: Tensor with moved dimensions
        """

        return self.movedim(t, source, destination)
    

    def permute(self, t: Tensor, *axes):
        """
        Method for changing order of axes of the input tensor

        Args:
            t (Tensor): Input tensor
            axes (Tuple[int]): Order of the axes of the input tensor the output tensor needs to be with

        Returns:
            Tensor: Tensor with changed order of axes
        """

        out = Tensor(np.transpose(t.data, axes), requires_grad=t.requires_grad, _children=(t,))

        def _backward():
            # during backward we need to transpose the gradient of the output tensor into the axes order of the input tensor
            if t.requires_grad:
                reversed_axes = np.argsort(axes)
                t.grad += np.transpose(out.grad, reversed_axes)

        out._backward = _backward

        return out


    def flatten(self, t: Tensor):
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


    def unflatten(self, t: Tensor, dim: int, *sizes):
        """
        Method for changing a tensor dimension into a new shape

        Args:
            t (Tensor): Input tensor
            dim (int): Dimension to unflatten
            sizes (Tuple[int]): Shape the dimension needs to be fit in

        Returns:
            Tensor: Output tensor with new shape
        """

        ndim = t.ndim
        shape = t.shape

        if not (-ndim <= dim < ndim):
            raise IndexError("index out of range")
        
        dim %= t.ndim

        new_shape = shape[:dim] + sizes + shape[dim + 1:]

        return self.reshape(t, *new_shape)


    def expand(self, t: Tensor, *shape):
        """
        Method for changing the input tensor to a larger shape

        Args:
            t (Tensor): Input tensor
            shape (Tuple[int]): Shape the input tensor needs to be expanded to

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


    def squeeze(self, t: Tensor, axis=None):
        """
        Method for removing axes of length 1 from the input tensor

        Args:
            t (Tensor): Input tensor
            axis (None or int or Tuple[int]): Axes with lenght 1 which need to be removed from the input tensor

        Returns:
            Tensor: Tensor with removed axes
        """

        out = Tensor(np.squeeze(t.data, axis=axis), requires_grad=t.requires_grad, _children=(t,))

        def _backward():
            if t.requires_grad:
                t.grad += out.grad.reshape(t.shape)

        out._backward = _backward

        return out


    def unsqueeze(self, t: Tensor, axis):
        """
        Method for adding new dimensions to the input tensor

        Args:
            t (Tensor): Input tensor
            axis (int or Tuple[int]): Positions where the new axes will be added

        Returns:
            Tensor: Tensor with added axes
        """

        out = Tensor(np.expand_dims(t.data, axis=axis), requires_grad=t.requires_grad, _children=(t,))

        def _backward():
            if t.requires_grad:
                t.grad += out.grad.reshape(t.shape)

        out._backward = _backward
        
        return out
    

    def split(self, t: Tensor, split_size_or_sections, dim: int = 0):
        """
        Method for splitting the tensor into multiple ones along one dimension

        Args:
            t (Tensor): Input tensor
            split_size_or_sections (int or Tuple[int]): Size of every chunk for splitting or range of sizes for all chunks
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
            if sum(split_size_or_sections) != t.shape[dim]:
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
    

    def chunk(self, t: Tensor, chunks: int, dim: int = 0):
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

        return self.split(t, chunk_size, dim)


    def cat(self, tensors: Tuple[Tensor], dim: int = 0):
        """
        Method for concatenating tuple of tensors into one tensor along one dimension

        Args:
            tensors (Tuple[Tensor]): Tuple of input tensors
            dim (int): Dimension the tensors are being concatenated along

        Returns:
            Tensor: Output tensor with view of concatenated initial tensors
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
    
    def stack(self, tensors: Tuple[Tensor], dim: int = 0):
        """
        Method for stacking tuple of tensors into one tensor along a new dimension

        Args:
            tensors (Tuple[Tensor]): Tuple of input tensors
            dim (int): New dimension the tensors are being stacked along

        Returns:
            Tensor: Output tensor with view of stacked initial tensors
        """

        unsqueezed_tensors = tuple(self.unsqueeze(tensor, axis=dim) for tensor in tensors)

        return self.cat(unsqueezed_tensors, dim)
    
    def vstack(self, tensors: Tuple[Tensor]):
        """
        Method for stacking tuple of tensors into one tensor along vertical dimension

        Args:
            tensors (Tuple[Tensor]): Tuple of input tensors

        Returns:
            Tensor: Output tensor with view of stacked initial tensors
        """

        reshaped_tensors = []
        
        # we need to stack along the vertical dimension H, so the tensors must have a shape at least (H, W)
        for tensor in tensors:
            # if a tensor have only one dimension (W,) we need to add dimension H to have necessary shape (1, W)
            if tensor.ndim == 1:
                reshaped_tensors.append(self.unsqueeze(tensor, axis=0))
            else:
                reshaped_tensors.append(tensor)

        return self.cat(tuple(reshaped_tensors), dim=0)
    
    def hstack(self, tensors: Tuple[Tensor]):
        """
        Method for stacking tuple of tensors into one tensor along horizontal dimension

        Args:
            tensors (Tuple[Tensor]): Tuple of input tensors

        Returns:
            Tensor: Output tensor with view of stacked initial tensors
        """

        # we need to stack along the horisontal dimension W, so a shape (W,) is enough for operation
        if tensors[0].ndim == 1:
            return self.cat(tensors, dim=0)
        else:
            return self.cat(tensors, dim=1)
    
    def dstack(self, tensors: Tuple[Tensor]):
        """
        Method for stacking tuple of tensors into one tensor along depth dimension

        Args:
            tensors (Tuple[Tensor]): Tuple of input tensors

        Returns:
            Tensor: Output tensor with view of stacked initial tensors
        """

        reshaped_tensors = []

        # we need to stack along the depth dimension D, so the tensors must have a shape at least (H, W, D)
        for tensor in tensors:
            # if a tensor have only one dimension (W,) we need to add dimensions H and D to have necessary shape (1, W, 1)
            if tensor.ndim == 1:
                reshaped_tensors.append(self.unsqueeze(self.unsqueeze(tensor, axis=0), axis=2))
            # if a tensor have only two dimensions (H, W) we need to add dimension D to have necessary shape (H, W, 1)
            elif tensor.ndim == 2:
                reshaped_tensors.append(self.unsqueeze(tensor, axis=2))
            else:
                reshaped_tensors.append(tensor)

        return self.cat(tuple(reshaped_tensors), dim=2)