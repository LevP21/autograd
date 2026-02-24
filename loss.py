from module import Module
from utils import clip

class MSELoss(Module):
    def forward(self, pred, target):
        diff = pred - target
        return (diff * diff).mean()

   
class CrossEntropyLoss(Module):
    def forward(self, pred, target):
        eps = 1e-8
        pred = clip(pred, eps, 1)
        return - (pred.log() * target).sum(axis=-1, keepdims=True)
    

class BCELoss(Module):
    def forward(self, pred, target):
        return - (pred.log() * target + (1 - pred).log() * (1 - target))