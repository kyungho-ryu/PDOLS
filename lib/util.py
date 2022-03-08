import numpy as np
from enum import IntEnum, unique
import torch.nn as nn
import torch

@unique
class Initialization(IntEnum) :
    DEFAULT=1
    XAVIER=2
    HE=3
    ORTHOGONAL=4

@unique
class Loss(IntEnum) :
    MSE=1
    SMOOTHL1=2

@unique
class Reward(IntEnum) :
    DEFAULT = 1
    SINGLEOBJ = 2
    MULTIOBJ = 3


def orthogonal_init(tensor, gain=1):
    '''
    Fills the input `Tensor` using the orthogonal initialization scheme from OpenAI
    Args:
        tensor: an n-dimensional `torch.Tensor`, where :math:`n \geq 2`
        gain: optional scaling factor

    Examples:
    #    >>> w = torch.empty(3, 5)
    #    >>> orthogonal_init(w)
    '''
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    u, s, v = torch.svd(flattened, some=True)
    if rows < cols:
        u.t_()
    q = u if tuple(u.shape) == (rows, cols) else v
    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor

def initNetworkWeight(_type, layer) :
    if _type == Initialization.DEFAULT.value :
        print("Network weight type", Initialization.DEFAULT.name)
    elif _type == Initialization.XAVIER.value :
        print("Network weight type", Initialization.XAVIER.name)
        nn.init.xavier_uniform_(layer.weight)
    elif _type == Initialization.HE.value:
        print("Network weight type", Initialization.HE.name)
        nn.init.kaiming_uniform_(layer.weight)
    elif _type == Initialization.ORTHOGONAL.value:
        if len(layer.weight.shape) >= 2:
            orthogonal_init(layer.weight, gain= 2**0.5)
        else:
            layer.weight.zero_()

def setDDPGActionSpace(actions) :
    # [0~ 2]
    actions = (actions +1)
    actions = np.clip(actions, 0, 2)
    return actions

def setPPOActionSpace(actions) :
    # [0~ 20]
    actions = np.clip(actions, -1, 1)
    #actions = actions +10
    #actions = np.clip(actions, 0, 20)
    return actions

def determinant(mat):
    '''
    Returns the determinant of a diagonal matrix
    Inputs:
    - mat, a diagonal matrix
    Returns:
    - The determinant of mat, aka product of the diagonal
    '''
    return torch.exp(torch.log(mat).sum())

def new_softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
