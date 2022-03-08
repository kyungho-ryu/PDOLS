import torch
from collections import deque

# UE 100, dataRate=26
class UE100_DR26() :
    def __init__(self):
        self.S = deque()
        self.S.appendleft(torch.tensor([2.6794, 2.8460]))
        self.S.appendleft(torch.tensor([2.3983, 2.8756]))
        self.S.appendleft(torch.tensor([2.8447, 1.5998]))

        self.W =
=