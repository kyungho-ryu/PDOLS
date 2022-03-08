import torch as th
t = th.tensor([1,2,3,4,5,6])
b = t.sum(-1)
print(b)