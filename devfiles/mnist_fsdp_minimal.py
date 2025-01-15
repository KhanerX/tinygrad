# model based off https://towardsdatascience.com/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
from typing import List, Callable
from tinygrad import Tensor, TinyJit, nn, GlobalCounters, Device
from tinygrad.helpers import getenv, trange
from tinygrad.nn.datasets import mnist
from tinygrad import Tensor, nn
from tinygrad.helpers import prod
from tinygrad.tensor import Function
from tinygrad.multi import MultiLazyBuffer
from tinygrad.tensor import all_tensors
from tinygrad.ops import buffers, all_metadata
from typing import Type, Callable
from tinygrad.tensor import _METADATA 

# Function to gather all the shards before a fwd and bwd, and scatter the gradients after bwd
class AllGather(Function):
  def forward(self, x:MultiLazyBuffer) -> MultiLazyBuffer:
    self.input_axis = x.axis
    self.bounds = x.bounds
    return x.all_gather()
  def backward(self, grad_output:MultiLazyBuffer) -> MultiLazyBuffer:
    rs = grad_output.scatter(self.input_axis, self.bounds)
    return rs

class Checkpoint(Function):
  @classmethod
  def apply(fxn:Type[Function], *x:Tensor, **kwargs) -> Tensor:
    ctx = fxn(x[0].device, *x, metadata=_METADATA.get())
    ctx.requires_grad = True

    ret = Tensor.__new__(Tensor)
    fn = kwargs.get('fn')

    ctx.fn = fn
    ctx.x = x[0]

    Tensor.no_grad = True
    ret = fn(x[0])
    Tensor.no_grad = False

    ret.requires_grad, ret.grad = ctx.requires_grad, None
    ret._ctx = ctx if ctx.requires_grad and not Tensor.no_grad else None  # used by autograd engine
    return ret

  def backward(self, grad_output):
    detached_input = Tensor(self.x.lazydata, device=self.x.device, requires_grad=True)
    detached_input._ctx = None
    self.fn(detached_input).backward(Tensor(grad_output, device=self.x.device, requires_grad=False))
    return detached_input.grad.lazydata;

# (temporary) Layer overrides to apply the AllGather in forward to the parameters that are going to be sharded
class FSDPLinear(nn.Linear):
  def __call__(self, x:Tensor) -> Tensor: return x.linear(
    AllGather.apply(self.weight),
  )

def checkpoint(x:Tensor, fn:Callable[[Tensor], Tensor]) -> Tensor:
  return Checkpoint.apply(x, fn=fn)

# Function to shard the parameters of the optimizer (including the model itself)
def fsdp(obj, devices: tuple[str]):
  for name, param in nn.state.get_state_dict(obj).items():
    print(f"\n {name} {param.dtype.itemsize} {param.shape} \n ")
    if(param.shape[0] == 1 or prod(param.shape) <= 1):
      param.to_(devices).realize()
    else:
      param.shard_(devices, axis=0).realize()
  return obj

class Model:
  def __init__(self):
    self.layers: List[Callable[[Tensor], Tensor]] = [
        FSDPLinear(2500, 2500, bias=False),
      ]

  def __call__(self, x:Tensor) -> Tensor: 
    x = x.flatten(1)
    for i, layer in enumerate(self.layers):
      print(f"\nLinear {i}, Input shape: {x.shape} \n ")
      x = layer(x)
      print(f"In Memory: {GlobalCounters.global_device_mem['CUDA'] //1000/1000:.1f} MB")
      print("---")
    
    return x

if __name__ == "__main__":
  Device.DEFAULT = "CUDA"
  #GPUS = tuple(f'{Device.DEFAULT}:{i}' for i in range(getenv("GPUS", 2)))
  GPUS = ("CLANG", "CUDA")

  Device.DEFAULT = "CLANG" #initialize the parameters on CPU
  print(f"\n Model \n ")
  model = Model()
  opt = fsdp(nn.optim.Adam(nn.state.get_parameters(model)), GPUS)
  print(f"\n End of modle init \n ")
  Device.DEFAULT = "CUDA"
  
  def train_step() -> Tensor:
    with Tensor.train():
      opt.zero_grad()
      Xt = Tensor.randint(128, 2500, requires_grad=False).shard_(GPUS, axis=0)
      print("Training Data")
      # TODO: this "gather" of samples is very slow. will be under 5s when this is fixed
      loss = model(Xt).sum()
      loss.backward()
      print(f"In Memory: {GlobalCounters.global_device_mem['CUDA'] //1000/1000:.1f} MB")
      for n, t in reversed(list(nn.state.get_state_dict(opt).items())):
        if(t.requires_grad):
          print(f"Gradient {n}: {t.grad.shape}")
          t.grad.realize()
          print(f"In Memory: {GlobalCounters.global_device_mem['CUDA'] //1000/1000:.1f} MB")
          print("---")
      return loss
    
  train_step()

