import os

from tensorflow.python.client import device_lib

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

print(device_lib.list_local_devices())
