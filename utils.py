import numpy as np

from tensor import Tensor


def clip(input_tensor, min, max):
    return Tensor(data=np.clip(input_tensor.data, min, max),
                  requires_grad=input_tensor.requires_grad,
                  _children=(input_tensor._previous))