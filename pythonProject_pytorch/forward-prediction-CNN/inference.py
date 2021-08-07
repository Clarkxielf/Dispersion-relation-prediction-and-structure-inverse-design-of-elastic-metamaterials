# import torch
# import torch.nn as nn
# from dataset import loader
# from model import CNN
# from torch.utils.tensorboard import SummaryWriter
# import os
# import numpy as np
# import matplotlib.pyplot as plt
#
# def predict_once_data(num):
#
#     DATA_DIR = os.getcwd()
#     weights = os.listdir(os.path.join(DATA_DIR, 'checkpoints'))
#     weights.sort()
#
#     weight = weights[num]
#
#     DIR = os.path.join(DATA_DIR, 'checkpoints/'+weight)
#     checkpoint = torch.load(DIR)
#
#     model = CNN(num_classes=31)
#
#     if torch.cuda.device_count() >= 1:
#       print("Let's use", torch.cuda.device_count(), "GPUs!")
#       model = nn.DataParallel(model)
#
#     model.cuda()
#
#     model.load_state_dict(checkpoint['state_dict'])
#
#     model.eval()
#     with torch.no_grad():
#
#         output_1_13 = []
#         label_1_13 = []
#
#         for batch_idx, (b_x, b_y) in enumerate(loader(num)[0]):
#             batch_test_output = model(b_x.unsqueeze(1).cuda())
#             output_1_13.append(batch_test_output.cpu().numpy()[:, None, :])
#             label_1_13.append(b_y.numpy()[:, None, :])
#
#         output_1_13 = np.concatenate(output_1_13, 0)
#         label_1_13 = np.concatenate(label_1_13, 0)
#
#     return output_1_13, label_1_13
#
#
#
#
#
# output_10_13 = []
# label_10_13 = []
# for num in range(0, 10):
#     output_1_13 = predict_once_data(num)[0]
#     label_1_13 = predict_once_data(num)[1]
#
#     output_10_13.append(output_1_13)
#     label_10_13.append(label_1_13)
#
# output_10_13 = np.concatenate(output_10_13, 1)
# label_10_13 = np.concatenate(label_10_13, 1)
#
# save_dir = 'results'
#
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
#
# import scipy.io as sio
#
# sio.savemat(save_dir+'/results.mat', {'output_10_13': output_10_13,
#                                       'label_10_13': label_10_13})  # 写入mat文件




import torch
import torch.nn as nn
from dataset import loader
from model import CNN
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

output_10_13 = sio.loadmat('results/results.mat')['output_10_13']
label_10_13 = sio.loadmat('results/results.mat')['label_10_13']

output_10_13 = np.concatenate([output_10_13[:, :, :10], output_10_13[:, :, 11:]], axis=-1)
label_10_13 = np.concatenate([label_10_13[:, :, :10], label_10_13[:, :, 11:]], axis=-1)

MAE_PER_AVG = 1-np.mean(np.abs(output_10_13-label_10_13)/label_10_13)

print(f'MAE_PER_AVG:{MAE_PER_AVG}')