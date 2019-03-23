# pip install torch torchvision xarray netCDF4

import os
import sys
import os.path as op
from math import ceil
from sys import stdout

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import xarray as xr

def print_progress(progression_str, current, end):
    stdout.write('\r')
    stdout.write(progression_str.format(current, end))
    stdout.flush()

if len(sys.argv) < 3:
    sys.stderr.write("Please provide input_path output_path [batch_size]")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]
batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 256

transform = transforms.Compose([
    lambda img: np.array(img)
])

trainset = torchvision.datasets.CIFAR10(root=input_path, train=True, # download=True,
                                        transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=input_path, train=False, # download=True,
                                       transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

train_container = xr.DataArray(np.zeros((10, 5000, 32, 32, 3), dtype='u1'))
test_container = xr.DataArray(np.zeros((10, 1000, 32, 32, 3), dtype='u1'))

train_batch_cnt = ceil(50000/batch_size)
test_batch_cnt = ceil(10000/batch_size)

progression_str = 'Creating training dataset: batch {:d}/{:d}'
stdout.write(progression_str.format(0, train_batch_cnt))
stdout.flush()

train_indices = [0] * 10
for batch_idx, (inputs, targets) in enumerate(trainloader):
    for idx, (input, target) in enumerate(zip(inputs, targets)):
        class_idx = target.item()
        img_idx = train_indices[class_idx]
        train_container[class_idx, img_idx] = input.cpu().numpy()
        train_indices[class_idx] += 1
    print_progress(progression_str, batch_idx+1, train_batch_cnt)

train_dataset = xr.Dataset()
train_dataset['data'] = train_container

if not op.exists(output_path):
    os.makedirs(output_path)
train_dataset.to_netcdf(f'{output_path}/train-{batch_size}.h5')

progression_str = 'Creating test dataset: batch {:d}/{:d}'
stdout.write('\n'+progression_str.format(0, test_batch_cnt))
stdout.flush()

test_indices = [0] * 10
for batch_idx, (inputs, targets) in enumerate(testloader):
    for idx, (input, target) in enumerate(zip(inputs, targets)):
        class_idx = target.item()
        img_idx = test_indices[class_idx]
        test_container[class_idx, img_idx] = input.cpu().numpy()
        test_indices[class_idx] += 1
    print_progress(progression_str, batch_idx+1, test_batch_cnt)
stdout.write('\n')
stdout.flush()

test_dataset = xr.Dataset()
test_dataset['data'] = test_container
test_dataset.to_netcdf(f'{output_path}/test-{batch_size}.h5')
