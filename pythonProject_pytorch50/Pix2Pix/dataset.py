import os
import scipy.io as sio
import numpy as np
import torch.utils.data as Data
import torch
import matplotlib.pyplot as plt

# BATCH_SIZE = 512
# SPLIT_RATIO = 0.8
#
# dataset = sio.loadmat(f'../shuffle_matrix_train_unique_20210510_classify.mat')['train']
# # dataset = sio.loadmat(f'shuffle_matrix_train_unique_20210510_classify.mat')['train']
# all_data = np.concatenate([[i.T for i in dataset[4]]], axis=0).astype('float32')   # N*10*31
# all_label = np.concatenate([[i for i in dataset[6]]], axis=0).astype('float32')   # N*50*50
#
# all_data = all_data.reshape(-1, 10, 1, 31)
# all_label = all_label.reshape(-1, 1, 50, 50)
#
# split_dataset = int(all_data.shape[0]*SPLIT_RATIO)
# Xtrain = torch.tensor(all_data[:split_dataset])
# Labeltrain = torch.tensor(all_label[:split_dataset])
# Xtest = torch.tensor(all_data[split_dataset:])
# Labeltest = torch.tensor(all_label[split_dataset:])
#
#
# # folder = '../save'
# # from torchvision.utils import save_image
# # all_label = Labeltest[-100:]
# # for idx in range(len(all_label)):
# #     save_image(all_label[idx], folder + f"/y_gen_{idx}_0.png")
#
#
# train_data = Data.TensorDataset(Xtrain, Labeltrain)
# train_loader = Data.DataLoader(dataset=train_data,
#                                batch_size=BATCH_SIZE,
#                                shuffle=True)
#
# test_data = Data.TensorDataset(Xtest, Labeltest)
# test_loader = Data.DataLoader(dataset=test_data,
#                               batch_size=1,
#                               shuffle=False)






# # dataset = sio.loadmat(f'../flat_band.mat')['flatband']
# dataset = sio.loadmat(f'flat_band.mat')['flatband']
# # all_data = np.concatenate([[i.T for i in dataset[2]]], axis=0).astype('float32')   # N*10*31
# all_data = dataset[2][0].T.astype('float32')[None, ...]
#
#
# all_label = np.random.random((1, 50, 50))
#
# all_data = all_data.reshape(-1, 10, 1, 31)
# all_label = all_label.reshape(-1, 1, 50, 50)
#
# Xtest = torch.tensor(all_data)
# Labeltest = torch.tensor(all_label)
#
# test_data = Data.TensorDataset(Xtest, Labeltest)
# test_loader = Data.DataLoader(dataset=test_data,
#                               batch_size=1,
#                               shuffle=False)






# folder = '../save'
# from torchvision.utils import save_image
# import torch
# import scipy.io as sio
#
# dataset = sio.loadmat(f'../inference_flat_band.mat')['pre_10_31']
#
# for index in range(len(dataset)):
#     data = dataset[index]
#     for i in range(len(data)):
#         plt.plot(data[i])
#
#     plt.savefig(f'{folder}/input_{index + 1}')
#     plt.show()




# from torchvision.utils import save_image
#
# dataset = sio.loadmat(f'../flat_band.mat')['flatband']
# dataset = torch.tensor(dataset[1][0].astype('float32'))
#
# save_image(dataset, f"../Y_1.png")



# dataset = sio.loadmat(f'../flat_band.mat')['flatband']
# dataset = dataset[2][0].T
# for i in range(len(dataset)):
#     plt.plot(dataset[i])
#
# plt.savefig(f'../Y_2')
# plt.show()


folder = '../save'
from torchvision.utils import save_image
import torch
import scipy.io as sio

datasets = sio.loadmat(f'../inference_flat_band.mat')['pre_10_31']

dataset = sio.loadmat(f'../flat_band.mat')['flatband']
dataset = dataset[2][0].T

for index in range(len(datasets)):
    data = datasets[index]
    # for i in range(len(data)):
    plt.plot(data[8])
    plt.plot(dataset[8])

    plt.savefig(f'{folder}/X_{index + 1}')
    plt.show()
