from tinygrad import Tensor
from tinygrad.helpers import GlobalCounters
import numpy as np


GPUS = ("CUDA", "CLANG")
a = Tensor(np.random.randn(1, 2500, 2500), requires_grad=True).realize().shard_(GPUS, axis=1)
print(a)
b = a.expand((512, 2500, 2500))
print(b)
b.sum().backward()
a.grad.realize()
print(a.grad)
print(GlobalCounters.global_device_mem)
print(GlobalCounters.global_ops)