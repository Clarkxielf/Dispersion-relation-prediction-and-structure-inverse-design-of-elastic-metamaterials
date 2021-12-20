import scipy.io as sio
import numpy as np
import torch.utils.data as Data
import torch

# dataset = sio.loadmat(f'../flat_band.mat')['flatband']
dataset = sio.loadmat(f'flat_band.mat')['flatband']
# all_data = np.concatenate([[i.T for i in dataset[2]]], axis=0).astype('float32')   # N*10*31
all_data = dataset[2][0].T.astype('float32')[None, ...]


all_label = np.random.random((1, 50, 50))

all_data = all_data.reshape(-1, 10, 1, 31)
all_label = all_label.reshape(-1, 1, 50, 50)

Xtest = torch.tensor(all_data)
Labeltest = torch.tensor(all_label)

test_data = Data.TensorDataset(Xtest, Labeltest)
test_loader = Data.DataLoader(dataset=test_data,
                              batch_size=1,
                              shuffle=False)


