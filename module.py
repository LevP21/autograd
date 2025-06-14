from abc import ABC, abstractmethod

import numpy as np

from tensor import Tensor


class Module(ABC):
    def parameters(self):
        return []


    def zero_grad(self):
        for param in self.parameters():
            param.grad = np.zeros_like(param.data)

    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    
    @abstractmethod
    def forward(self):
        pass


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers


    def parameters(self):
        params = []

        for layer in self.layers:
            if isinstance(layer, Module):
                params.extend(layer.parameters())

        return params


    def zero_grad(self):
        for layer in self.layers:
            if isinstance(layer, Module):
                layer.zero_grad()


    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x


class Linear(Module):
    def __init__(self, in_features, out_features):
        scale = np.sqrt(2 / in_features)
        self.W = Tensor(np.random.randn(in_features, out_features) * scale, requires_grad=True)
        self.b = Tensor(np.zeros(out_features), requires_grad=True)


    def parameters(self):
        return [self.W, self.b]
    

    def forward(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        return x @ self.W + self.b
