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
                t.grad += np.broadcast_to(grad, t.data.shape)

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
                        mean_count = sum((t.data.shape[ax] for ax in axis))
                    else:
                        mean_count = t.data.shape[axis]
                else:
                    mean_count = sum(t.data.shape)
                t.grad += np.broadcast_to(grad, t.data.shape) / mean_count

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
    

    def view(self, t: Tensor, *shape):
        """
        Method for changing the contiguous input tensor into a different shape

        Args:
            t (Tensor): Input tensor
            shape (Tuple(int)): Shape the input tensor needs to be fit in
        
        Returns:
            Tensor: Tensor with changed shape
        """

        if not t.is_contiguous():
            raise RuntimeError(
                "view requires contiguous tensor"
            )

        out = Tensor(t.data.reshape(*shape, copy=False), requires_grad=t.requires_grad, _children=(t,))

        def _backward():
            # during backward we need to reshape the gradient of the output tensor into the shape of the input tensor
            if t.requires_grad:
                t.grad += out.grad.reshape(t.data.shape)

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
            shape (Tuple(int)): Shape the input tensor needs to be fit in
        
        Returns:
            Tensor: Tensor with changed shape
        """

        try:
            return self.view(t, *shape)
        except RuntimeError:
            return self.view(t.contiguous(), *shape)
        

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


    def permute(self, t: Tensor, *axes):
        """
        Method for changing order of axes of the input tensor

        Args:
            t (Tensor): Input tensor
            axes (Tuple(int)): Order of the axes of the input tensor the output tensor needs to be with

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

        # saving initial shape for reshaping back the gradient
        original_shape = t.data.shape

        out = Tensor(t.data.flatten(), requires_grad=t.requires_grad, _children=(t,))

        def _backward():
            if t.requires_grad:
                t.grad += out.grad.reshape(original_shape)

        out._backward = _backward

        return out


    def expand(self, t: Tensor, *shape):
        """
        Method for changing the input tensor to a larger shape

        Args:
            t (Tensor): Input tensor
            shape (Tuple(int)): Shape the input tensor needs to be expanded to

        Returns:
            Tensor: Tensor with expanded shape
        """

        out = Tensor(np.broadcast_to(t.data, shape), requires_grad=t.requires_grad, _children=(t,))

        def _backward():
            # during backward we need to change back the gradient shape into the smaller shape of the input tensor
            if t.requires_grad:
                t.grad += unbroadcast(out.grad, t.data.shape)

        out._backward = _backward

        return out


    def squeeze(self, t: Tensor, axis=None):
        """
        Method for removing axes of length 1 from the input tensor

        Args:
            t (Tensor): Input tensor
            axis (None or int or Tuple(int)): Axes with lenght 1 which need to be removed from the input tensor

        Returns:
            Tensor: Tensor with removed axes
        """

        # saving initial shape for reshaping back the gradient
        original_shape = t.data.shape

        out = Tensor(np.squeeze(t.data, axis=axis), requires_grad=t.requires_grad, _children=(t,))

        def _backward():
            if t.requires_grad:
                t.grad += out.grad.reshape(original_shape)

        out._backward = _backward

        return out


    def unsqueeze(self, t: Tensor, axis):
        """
        Method for adding new dimensions to the input tensor

        Args:
            t (Tensor): Input tensor
            axis (int or Tuple(int)): Positions where the new axes will be added

        Returns:
            Tensor: Tensor with added axes
        """

        out = Tensor(np.expand_dims(t.data, axis=axis), requires_grad=t.requires_grad, _children=(t,))

        def _backward():
            if t.requires_grad:
                t.grad += out.grad.reshape(t.data.shape)

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
