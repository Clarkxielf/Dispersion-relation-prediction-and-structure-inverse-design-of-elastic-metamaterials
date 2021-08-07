import os
import scipy.io as sio
import numpy as np
import torch.utils.data as Data
import torch
import matplotlib.pyplot as plt
import random


MAT_NAME = 'matrix_train_unique_20210510_classify.mat'

dataset = sio.loadmat(MAT_NAME)['exist']

shuffle_idx = list(range(dataset.shape[-1]))
random.shuffle(shuffle_idx)

all_data = dataset[:, shuffle_idx]

sio.savemat('shuffle_matrix_train_unique_20210510_classify.mat', {'train': all_data})

