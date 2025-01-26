from typing import List, Callable
from tinygrad import Tensor, nn, GlobalCounters, Device
from tinygrad.helpers import prod

def fsdp(obj, devices: tuple[str]):
  for name, param in nn.state.get_state_dict(obj).items():
    print(f"\n {name} {param.dtype.itemsize} {param.shape} \n ")
    if(param.shape[0] == 1 or prod(param.shape) <= 1):
      param.to_(devices)
    else:
      param.shard_(devices, axis=0)
  return obj

class Model:
  def __init__(self):
    self.layers: List[Callable[[Tensor], Tensor]] = [
      nn.Linear(2500, 2500, bias=False),
      nn.Linear(2500, 2500, bias=False),
      nn.Linear(2500, 2500, bias=False),
      nn.Linear(2500, 2500, bias=False),
      nn.Linear(2500, 2500, bias=False),
      nn.Linear(2500, 2500, bias=False),
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
  GPUS = ("CLANG", "CUDA")
  print(f"\n Model \n ")
  model = Model()
  opt = fsdp(nn.optim.Adam(nn.state.get_parameters(model)), GPUS)
  for param in opt.params:
    param.lazydata.placement = "replicate"
  print(f"\n End of modle init \n ")
  Device.DEFAULT = "CUDA"
  
  def train_step() -> Tensor:
    with Tensor.train():
      opt.zero_grad()
      Xt = Tensor.randint(128, 2500, requires_grad=False).shard_(GPUS, axis=0)
      print("Training Data")
      # TODO: this "gather" of samples is very slow. will be under 5s when this is fixed
      loss = model(Xt).sum()
      loss.backward(retain_graph=True)
      print(f"In Memory: {GlobalCounters.global_device_mem['CUDA'] //1000/1000:.1f} MB")
      opt.step()
      return loss

  for i in range(10):
    train_step()
    train_step()
    train_step()
