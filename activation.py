from module import Module


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class Sigmoid(Module):
    def forward(self, x):
        return (1 / (1 + (-x).exp()))
    

class Softmax(Module):
    def forward(self, x):
        return x.exp() / x.exp().sum(axis=-1, keepdims=True)