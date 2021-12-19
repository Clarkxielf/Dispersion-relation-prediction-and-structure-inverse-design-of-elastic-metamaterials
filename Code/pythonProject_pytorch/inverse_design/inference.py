import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import save_image
import os
import scipy.io as sio
import numpy as np
import torch.utils.data as Data
from skimage.measure._label import label
import cv2 as cv

import cGAN.config as config
from cGAN.dataset import test_loader
from cGAN.generator_model import Generator
from CNN.model import CNN


BATCH_SIZE = 512

def cv_imread(filePath):

    cv_img = cv.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    cv_img = cv.cvtColor(cv_img, cv.COLOR_RGB2BGR)

    return cv_img

def threshold_image(image):

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 255, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)

    return binary

def inference_some_examples(gen, x, y, idx, epoch, folder):

    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)

        save_image(y_fake, folder + f"/y_gen_0_{idx}_{epoch}.png")

        y_fake += torch.flip(y_fake, (2,))   # row
        y_fake += torch.flip(y_fake, (3,))   # column
        y_fake += y_fake.transpose(-1, -2)   # diagonal

        save_image(y_fake, folder + f"/y_gen_1_{idx}_{epoch}.png")

        if epoch==1:
            save_image(y, folder + f"/input_{idx}_{epoch}.png")


        y_fake_name = folder + f"/y_gen_1_{idx}_{epoch}.png"
        y_fake_image = cv_imread(y_fake_name)
        y_fake_image = cv.medianBlur(y_fake_image, ksize=3)
        y_binary = threshold_image(image=y_fake_image)

        labels, num = label(y_binary, return_num=True, background=255)

        max_area = 0
        value = 0
        for i in range(1, num + 1):
            value_area = (np.ones_like(labels) * i == labels).sum()

            if value_area > max_area:
                max_area = value_area
                value = i

        y_binary = 1 - np.array(np.ones_like(labels) * value == labels).astype('uint8')

        y_fake = torch.from_numpy(y_binary)[None, None, ...]
        y_fake = torch.clamp(y_fake.float(), min=0., max=1.)

        save_image(y_fake, folder + f"/y_gen_1_{idx}_{epoch}_binary.png")

        return y, y_fake

def save_some_examples(gt, test, epoch, folder):

    save_image(gt, folder + f"/input_{epoch}.png")
    save_image(test, folder + f"/y_gen_{epoch}.png")



def loader(pre_50_50, label_10_31):

    pre_data = Data.TensorDataset(pre_50_50, label_10_31)
    pre_loader = Data.DataLoader(dataset=pre_data,
                                  batch_size=BATCH_SIZE,
                                  shuffle=False)

    return pre_loader

def predict_once_data(num, pre_loader):

    DATA_DIR = os.getcwd()
    weights = os.listdir(os.path.join(DATA_DIR, 'CNN/checkpoints'))
    weights.sort()

    weight = weights[num]

    DIR = os.path.join(DATA_DIR, 'CNN/checkpoints/'+weight)
    checkpoint = torch.load(DIR)

    model = CNN(num_classes=31)

    if torch.cuda.device_count() >= 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      model = nn.DataParallel(model)

    model.cuda()

    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    with torch.no_grad():

        pre_1_13 = []

        for batch_idx, (b_x, b_y) in enumerate(pre_loader):
            batch_test_output = model(b_x.unsqueeze(1).cuda())
            pre_1_13.append(batch_test_output.cpu().numpy()[:, None, :])

        pre_1_13 = np.concatenate(pre_1_13, 0)

    return pre_1_13



gen = Generator(in_channels=1, features=32).to(config.DEVICE)
gen.load_state_dict(torch.load('cGAN/checkpoints/gen_500.pth.tar')['state_dict'], strict=False)

loop = tqdm(test_loader, leave=True)

pre_50_50 = []
label_50_50 = []
label_10_31 = []
noise_label_10_31 = []
noise_10_31 = []
num_choice_idx = 1
num_noise = 2000

# choice_idx = np.random.choice(len(test_loader), num_choice_idx, replace=False)
# choice_idx = list(range(len(test_loader)))[-num_choice_idx:]
choice_idx = list(range(len(test_loader)))[:]


log = open(os.path.join('inference/log.txt'), 'a')

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

print_log(f'choice_idx:\n{choice_idx}', log)

for idx, (x0, y) in enumerate(loop):

    if idx in choice_idx:
        for num_iter in range(num_noise):
            if num_iter!=0:

                # noise = torch.randn([*x0.shape])
                # x = x0 + (noise.normal_(mean=0, std=1).clamp_(min=-1, max=1) / 25)

                # x = x0 + (torch.randn([*x0.shape])/5)*(torch.softmax(x0.max(-1).values, 1)[..., None])

                noise = torch.randn([*x0.shape])/50
                x = x0 + noise

            else:
                noise = torch.zeros([*x0.shape]) / 50
                x = x0

            y, y_fake = inference_some_examples(gen, x.to(config.DEVICE), y.to(config.DEVICE), idx, num_iter+1, folder='inference')   # N*1*50*50

            label_50_50.append(y.squeeze(1))   # N*50*50
            pre_50_50.append(y_fake.squeeze(1))
            label_10_31.append(x0.reshape(-1, 10, 31).to(config.DEVICE))
            noise_label_10_31.append(x.reshape(-1, 10, 31).to(config.DEVICE))
            noise_10_31.append(noise.reshape(-1, 10, 31).to(config.DEVICE))


label_50_50 = torch.cat(label_50_50, dim=0)
pre_50_50 = torch.cat(pre_50_50, dim=0)
label_10_31 = torch.cat(label_10_31, dim=0)
noise_label_10_31 = torch.cat(noise_label_10_31, dim=0)
noise_10_31 = torch.cat(noise_10_31, dim=0)


pre_10_31 = []
pre_loader = loader(pre_50_50, label_10_31)

for num in range(0, 10):

    pre_1_13 = predict_once_data(num, pre_loader)

    pre_10_31.append(pre_1_13)

pre_10_31 = np.concatenate(pre_10_31, 1)
label_10_31 = label_10_31.cpu().numpy()
noise_label_10_31 = noise_label_10_31.cpu().numpy()
noise_10_31 = noise_10_31.cpu().numpy()

# pre_10_31_MAE = np.concatenate([pre_10_31[:, 0, :10], pre_10_31[:, 0, 11:]], axis=-1)
# label_10_31_MAE = np.concatenate([label_10_31[:, 0, :10], label_10_31[:, 0, 11:]], axis=-1)
# pre_10_31_MAE = pre_10_31[:, 0, :10]
# label_10_31_MAE = label_10_31[:, 0, :10]
pre_10_31_MAE = pre_10_31[:, 8, :]
label_10_31_MAE = label_10_31[:, 8, :]

MAE_PER_AVG = np.mean(np.abs(pre_10_31_MAE-label_10_31_MAE)/label_10_31_MAE, axis=(-1))

MAE_PER_AVG = MAE_PER_AVG.reshape(num_choice_idx, num_noise)


num_save = 5
top5_idx = np.argpartition(MAE_PER_AVG, num_save)[:, :num_save]
top5_idx = top5_idx + (np.array(list(range(num_choice_idx)))*num_noise).reshape(-1, 1)
top5_idx = top5_idx.flatten()


label_50_50 = label_50_50[top5_idx]
pre_50_50 = pre_50_50[top5_idx]
label_10_31 = label_10_31[top5_idx]
pre_10_31 = pre_10_31[top5_idx]
noise_label_10_31 = noise_label_10_31[top5_idx]
noise_10_31 = noise_10_31[top5_idx]


symmetry_idx = pre_50_50[:, :, -1].sum(-1)!=1*50

label_50_50 = label_50_50[symmetry_idx]
pre_50_50 = pre_50_50[symmetry_idx]
label_10_31 = label_10_31[symmetry_idx]
pre_10_31 = pre_10_31[symmetry_idx]
noise_label_10_31 = noise_label_10_31[symmetry_idx]
noise_10_31 = noise_10_31[symmetry_idx]


for i in range(len(label_50_50)):
    save_some_examples(gt=label_50_50[i], test=pre_50_50[i], epoch=i+1, folder='save')


sio.savemat('inference_flat_band.mat', {'label_10_31': label_10_31,
                              'label_50_50': label_50_50.cpu().numpy(),
                              'pre_50_50': pre_50_50.cpu().numpy(),
                              'pre_10_31': pre_10_31,
                              'noise_10_31': noise_10_31,
                              'noise_label_10_31': noise_label_10_31})