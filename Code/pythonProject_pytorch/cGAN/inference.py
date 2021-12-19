import torch
from dataset import test_loader
from generator_model import Generator
from tqdm import tqdm
import config
from torchvision.utils import save_image
import numpy as np
import cv2 as cv
from skimage.measure._label import label

def cv_imread(filePath):

    cv_img = cv.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    cv_img = cv.cvtColor(cv_img, cv.COLOR_RGB2BGR)

    return cv_img

def threshold_image(image):

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 255, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)

    return binary

def save_some_examples(gen, x, y, epoch, folder):

    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)

        y_fake += torch.flip(y_fake, (2,))   # row
        y_fake += torch.flip(y_fake, (3,))   # column
        y_fake += y_fake.transpose(-1, -2)   # diagonal

        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(y, folder + f"/input_{epoch}.png")

        y_fake_name = folder + f"/y_gen_{epoch}.png"
        y_fake_image = cv_imread(y_fake_name)
        y_fake_image = cv.medianBlur(y_fake_image, ksize=3)
        y_binary = threshold_image(image=y_fake_image)

        labels, num = label(y_binary, return_num=True, background=255)

        max_area = 0
        for i in range(1, num + 1):
            value_area = (np.ones_like(labels) * i == labels).sum()

            if value_area > max_area:
                max_area = value_area
                value = i

        y_binary = 1 - np.array(np.ones_like(labels) * value == labels).astype('uint8')

        y_fake = torch.from_numpy(y_binary)[None, None, ...]
        y_fake = torch.clamp(y_fake.float(), min=0., max=1.)

        save_image(y_fake, folder + f"/y_gen_{epoch}_binary.png")

        return y, y_fake


gen = Generator(in_channels=1, features=32).to(config.DEVICE)

gen.load_state_dict(torch.load('checkpoints/gen_500.pth.tar')['state_dict'], strict=False)


loop = tqdm(test_loader, leave=True)

output_50_50 = []
label_50_50 = []
label_10_31 = []
for idx, (x, y) in enumerate(loop):

    if idx>14120:

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