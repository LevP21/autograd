import numpy as np


def unbroadcast(grad, shape):
    if grad.shape == ():
        grad = np.full(shape, grad)

    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)

    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


class Tensor:
    def __init__(self, data, requires_grad=False, _children=()):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad

        self.shape = self.data.shape
        self.T = self.data.T
        self._backward = lambda: None
        self._prev = set(_children)

    
    def item(self):
        return self.data


    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other))

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad, self.grad.shape)
            if other.requires_grad:
                other.grad += unbroadcast(out.grad, other.grad.shape)

        out._backward = _backward
        return out


    def __radd__(self, other):
        return self + other


    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other))

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad, self.grad.shape)
            if other.requires_grad:
                other.grad -= unbroadcast(out.grad, other.grad.shape)

        out._backward = _backward
        return out


    def __rsub__(self, other):
        return self - other


    def __neg__(self):
        out = Tensor(-self.data, requires_grad=self.requires_grad, _children=(self,))

        def _backward():
            if self.requires_grad:
                self.grad += -out.grad

        out._backward = _backward
        return out


    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other))

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(other.data, self.grad.shape) * unbroadcast(out.grad, self.grad.shape)
            if other.requires_grad:
                other.grad += unbroadcast(self.data, other.grad.shape) * unbroadcast(out.grad, other.grad.shape)

        out._backward = _backward
        return out


    def __rmul__(self, other):
        return self * other


    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other))

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast((1 / other.data), self.grad.shape) * unbroadcast(out.grad, self.grad.shape)
            if other.requires_grad:
                other.grad += unbroadcast((-self.data / (other.data ** 2)), other.grad.shape) * unbroadcast(out.grad, other.grad.shape)

        out._backward = _backward
        return out


    def __rtruediv__(self, other):
        return self / other


    def __pow__(self, power):
        assert isinstance(power, (int, float))
        out = Tensor(self.data ** power, requires_grad=self.requires_grad, _children=(self,))

        def _backward():
            if self.requires_grad:
                self.grad += (power * self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out
    

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other))

        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out
    

    def sum(self, axis=None, keepdims=False):
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,))

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if not keepdims and axis is not None:
                    grad = np.expand_dims(grad, axis=axis)
                self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out


    def mean(self, axis=None, keepdims=False):
        out_data = np.mean(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,))

        def _backward():
            if self.requires_grad:
                grad = out.grad / np.prod(self.data.shape if axis is None else self.data.shape[axis])
                self.grad += unbroadcast(grad, self.data.shape)

        out._backward = _backward
        return out
    

    def relu(self):
        out_data = np.maximum(self.data, 0.0)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,))

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (self.data > 0)

        out._backward = _backward
        return out


    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad, _children=(self,))

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out


    def permute(self, *axes):
        out = Tensor(np.transpose(self.data, axes), requires_grad=self.requires_grad, _children=(self,))

        def _backward():
            if self.requires_grad:
                reversed_axes = np.argsort(axes)
                self.grad += np.transpose(out.grad, reversed_axes)

        out._backward = _backward
        return out


    def flatten(self):
        original_shape = self.data.shape
        out = Tensor(self.data.flatten(), requires_grad=self.requires_grad, _children=(self,))

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(original_shape)

        out._backward = _backward
        return out


    def expand(self, *shape):
        out = Tensor(np.broadcast_to(self.data, shape), requires_grad=self.requires_grad, _children=(self,))

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad, self.data.shape)

        out._backward = _backward
        return out


    def squeeze(self, axis=None):
        original_shape = self.data.shape
        out = Tensor(np.squeeze(self.data, axis=axis), requires_grad=self.requires_grad, _children=(self,))

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(original_shape)

        out._backward = _backward
        return out


    def unsqueeze(self, axis):
        out = Tensor(np.expand_dims(self.data, axis), requires_grad=self.requires_grad, _children=(self,))

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out


    def backward(self):
        graph = []
        visited = set()

        def build_graph(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for child in tensor._prev:
                    build_graph(child)
                graph.append(tensor)
        build_graph(self)

        for tensor in reversed(graph):
            tensor._backward()


    def __repr__(self):
        return f"Tensor(data={self.data}\nGrad={self.grad})"
