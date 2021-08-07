import torch
import torch.nn as nn
import torch.optim as optim
import config
from dataset import train_loader, test_loader
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
import torch
import config
from torchvision.utils import save_image
import numpy as np

def save_some_examples(gen, x, y, epoch, folder):

    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake  # remove normalization#

        y_fake += torch.flip(y_fake, (2,))   # row
        y_fake += torch.flip(y_fake, (3,))   # column
        y_fake += y_fake.transpose(-1, -2)   # diagonal

        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(y, folder + f"/input_{epoch}.png")

        return y, y_fake


gen = Generator(in_channels=1, features=32).to(config.DEVICE)

gen.load_state_dict(torch.load('./checkpoints/Pix2Pix_Colorize_Anime/gen_500.pth.tar')['state_dict'])

gen.cuda()

loop = tqdm(test_loader, leave=True)

output_50_50 = []
label_50_50 = []
label_10_31 = []
for idx, (x, y) in enumerate(loop):

    if idx>700 and idx<800:

        y, y_fake = save_some_examples(gen, x, y, idx+1, folder="inference")

        label_50_50.append(y.squeeze(1).cpu().numpy())
        output_50_50.append(y_fake.squeeze(1).cpu().numpy())
        label_10_31.append(x.reshape(-1, 10, 31).cpu().numpy())

label_10_31 = np.concatenate(label_10_31, axis=0)
label_50_50 = np.concatenate(label_50_50, axis=0)
output_50_50 = np.concatenate(output_50_50, axis=0)

import scipy.io as sio

sio.savemat('inference.mat', {'label_10_31': label_10_31,
                              'label_50_50': label_50_50,
                              'output_50_50': output_50_50})