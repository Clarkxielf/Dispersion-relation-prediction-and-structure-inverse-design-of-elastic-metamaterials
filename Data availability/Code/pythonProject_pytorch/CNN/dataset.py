import scipy.io as sio
import numpy as np
import torch.utils.data as Data
import torch

def loader(num):

    SPLIT_RATIO = 0.8
    BATCH_SIZE = 128

    MAT_NAME = 'unit_cell_data.mat'

    dataset = sio.loadmat(MAT_NAME)['train']

    all_data = np.concatenate([i[None, ...] for i in dataset[6]], axis=0).astype('float32')   #N*50*50
    all_label = np.concatenate([[i.T[num] for i in dataset[4]]], axis=0).astype('float32')   #N*31

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
                                  batch_size=BATCH_SIZE,
                                  shuffle=False)

    return test_loader, train_loader
