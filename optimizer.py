from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr


    def zero_grad(self):
        for param in self.parameters:
            if param.requires_grad:
                param.grad = np.zeros_like(param.grad)


    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def step(self):
        for param in self.parameters:
            if param.requires_grad:
                param.data -= self.lr * param.grad