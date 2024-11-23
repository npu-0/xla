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
