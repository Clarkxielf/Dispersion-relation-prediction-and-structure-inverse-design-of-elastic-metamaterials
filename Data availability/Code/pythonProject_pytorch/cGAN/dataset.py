import scipy.io as sio
import numpy as np
import torch.utils.data as Data
import torch

BATCH_SIZE = 512
SPLIT_RATIO = 0.8

dataset = sio.loadmat(f'unit_cell_with_bandgap.mat')['train']

all_data = np.concatenate([[i.T for i in dataset[4]]], axis=0).astype('float32')   # N*10*31
all_label = np.concatenate([[i for i in dataset[6]]], axis=0).astype('float32')   # N*50*50

all_data = all_data.reshape(-1, 10, 1, 31)
all_label = all_label.reshape(-1, 1, 50, 50)


split_dataset = int(all_data.shape[0]*SPLIT_RATIO)
Xtrain = torch.tensor(all_data[:split_dataset])
Labeltrain = torch.tensor(all_label[:split_dataset])
Xtest = torch.tensor(all_data[split_dataset:])
Labeltest = torch.tensor(all_label[split_dataset:])


train_data = Data.TensorDataset(Xtrain, Labeltrain)
train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=BATCH_SIZE,
                               shuffle=True)

test_data = Data.TensorDataset(Xtest, Labeltest)
test_loader = Data.DataLoader(dataset=test_data,
                              batch_size=1,
                              shuffle=False)
