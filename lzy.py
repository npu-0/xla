import torch
import torch_xla
import torch_xla.core.xla_model as xm

dev = xm.xla_device()

x1 = torch.rand((3, 3)).to(dev)
x2 = torch.rand((3, 8)).to(dev)

y1 = torch.einsum('bs,st->bt', x1, x2)
print(torch_xla._XLAC._get_xla_tensors_text([y1]))

y1 = y1 + x2
print(torch_xla._XLAC._get_xla_tensors_text([y1]))

xm.mark_step()
print(torch_xla._XLAC._get_xla_tensors_text([y1]))

def dummy_step(x, y, loss, acc=False):
  z = torch.einsum('bs,st->bt', y, x)
  step_loss = z.sum().view(1,)
  if acc:
    loss = torch.cat((loss, step_loss))
  else:
    loss = step_loss
  xm.mark_step()
  return loss


import time
def measure_time(acc=False):
  exec_times = []
  iter_count = 1
  x = torch.rand((512, 8)).to(dev)
  y = torch.rand((512, 512)).to(dev)
  loss = torch.zeros(1).to(dev)
  for i in range(iter_count):
    tic = time.time()
    loss = dummy_step(x, y, loss, acc=acc)
    toc = time.time()
    exec_times.append(toc - tic)
  return exec_times

dyn = measure_time(acc=True) # acc= True Results in dynamic graph
st = measure_time(acc=False) # Static graph, computation shape, inputs and output shapes don't change
