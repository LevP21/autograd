from module import Module


class MSE(Module):
    def forward(self, pred, target):
        diff = pred - target
        return (diff * diff).mean()