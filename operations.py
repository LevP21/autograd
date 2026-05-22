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
