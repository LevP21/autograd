import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, _children=()):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad

        self.shape = data.shape
        self._backward = lambda: None
        self._prev = set(_children)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other))

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad

        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other))

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad -= out.grad

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
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad

        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other))

        def _backward():
            if self.requires_grad:
                self.grad += (1 / other.data) * out.grad
            if other.requires_grad:
                other.grad += (-self.data / (other.data ** 2)) * out.grad

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

    def backward(self):
        graph = []
        visited = set()

        def build_graph(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_graph(child)
                graph.append(t)
        build_graph(self)

        self.grad = np.ones_like(self.data)

        for t in reversed(graph):
            t._backward()

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
