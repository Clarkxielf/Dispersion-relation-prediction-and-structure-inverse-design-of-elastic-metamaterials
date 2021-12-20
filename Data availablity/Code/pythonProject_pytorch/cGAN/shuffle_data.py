import scipy.io as sio
import random


MAT_NAME = 'unit_cell_with_bandgap.mat'

dataset = sio.loadmat(MAT_NAME)['exist']

shuffle_idx = list(range(dataset.shape[-1]))
random.shuffle(shuffle_idx)

all_data = dataset[:, shuffle_idx]

sio.savemat('unit_cell_with_bandgap.mat', {'train': all_data})

