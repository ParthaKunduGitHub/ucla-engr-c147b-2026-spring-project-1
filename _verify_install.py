import torch
print("PyTorch", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

import torchvision
print("torchvision", torchvision.__version__)

import numpy
print("numpy", numpy.__version__)

import matplotlib
print("matplotlib", matplotlib.__version__)

try:
    import jupyterlab
    print("jupyterlab", jupyterlab.__version__)
except ImportError:
    print("jupyterlab: NOT INSTALLED")

try:
    import ipykernel
    print("ipykernel", ipykernel.__version__)
except ImportError:
    print("ipykernel: NOT INSTALLED")
