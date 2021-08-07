import torch
import torch.nn as nn
from dataset import loader
from model import CNN
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import matplotlib.pyplot as plt


def train_once_data(num):
    Iter = 1234

    torch.manual_seed(Iter)
    EPOCH = 230
    LR = 0.1

    model = CNN(num_classes=31)

    if torch.cuda.device_count() >= 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      model = nn.DataParallel(model)

    model.cuda()

    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-5)

    save_dir = os.path.join(os.getcwd(), f'checkpoint{num}')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log = open(os.path.join(save_dir, 'log.txt'), 'a')

    def print_log(print_string, log):
        print("{}".format(print_string))
        log.write('{}\n'.format(print_string))
        log.flush()

    def save_checkpoint(state, is_best, epoch):
        """
        Save the training model
        """
        if is_best:
            max_file_num = 0
            filelist = os.listdir(save_dir)
            filelist.sort()
            file_num = 0
            to_move = []
            for file in filelist:
                if 'model' in file:
                    file_num = file_num + 1
                    to_move.append(os.path.join(save_dir, file))
            if file_num > max_file_num:
                to_move.sort()
                os.remove(to_move[0])
                to_move.pop(0)

            torch.save(state, save_dir + (f'/model_best_{epoch}.pth.tar'))

    def adjust_learning_rate(optimizer, epoch, start_lr):
        lr = start_lr * (0.1 ** (epoch // 100))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    BEST_test_MAE_PER_AVG = np.inf
    writer = SummaryWriter(f'{save_dir}/logs/CNN')
    START_EPOCH = 0

    # DATA_DIR = os.getcwd()
    # checkpoint = torch.load(os.path.join(DATA_DIR, 'checkpoint/model_best_87.pth.tar'))
    # model.load_state_dict(checkpoint['state_dict'])
    # START_EPOCH = checkpoint['epoch']
    # test_BEST_MAE_PER_AVG = checkpoint['test_BEST_MAE_PER_AVG']
    # opt.load_state_dict(checkpoint['optimizer'])

    for epoch in range(START_EPOCH, EPOCH):

        adjust_learning_rate(opt, epoch, LR)

        model.train()
        train_MAE = 0.
        train_MAE_PER = 0.
        for batch_idx, (b_x, b_y) in enumerate(loader(num)[1]):

            output = model(b_x.unsqueeze(1).cuda())

            MAE = torch.mean(torch.abs(output-b_y.cuda()))

            opt.zero_grad()
            MAE.backward()
            opt.step()

            train_MAE += MAE.item()

            b_y = torch.cat([b_y[:, :10], b_y[:, 11:]], dim=-1)
            output = torch.cat([output[:, :10], output[:, 11:]], dim=-1)
            train_MAE_PER += torch.mean(torch.abs(output-b_y.cuda())/b_y.cuda()).item()


        train_MAE_AVG = train_MAE/(batch_idx+1)
        train_MAE_PER_AVG = train_MAE_PER/(batch_idx+1)

        model.eval()
        test_MAE = 0.
        test_MAE_PER = 0.
        with torch.no_grad():
            for batch_idx, (b_x, b_y) in enumerate(loader(num)[0]):
                test_output = model(b_x.unsqueeze(1).cuda())

                test_MAE += torch.mean(torch.abs(test_output-b_y.cuda())).item()

                b_y = torch.cat([b_y[:, :10], b_y[:, 11:]], dim=-1)
                test_output = torch.cat([test_output[:, :10], test_output[:, 11:]], dim=-1)
                test_MAE_PER += torch.mean(torch.abs(test_output-b_y.cuda())/b_y.cuda()).item()

            test_MAE_AVG = test_MAE/(batch_idx+1)
            test_MAE_PER_AVG = test_MAE_PER/(batch_idx+1)

            # x = np.arange(0, len(b_y[0])) + 1
            # plt.plot(x, b_y[0].numpy(), 'r-', label='b_y')
            # plt.plot(x, test_output[0].cpu().numpy(), 'b--', label='test_output')
            # plt.legend()
            # plt.grid()
            # plt.show()

            print_log(f'Epoch: {epoch}\t'
                      f'|train_MAE_AVG: {train_MAE_AVG:.8f}\t|train_MAE_PER_AVG: {train_MAE_PER_AVG:.8f}\t'
                      f'|test_MAE_AVG: {test_MAE_AVG:.8f}\t|test_MAE_PER_AVG: {test_MAE_PER_AVG:.8f}', log)

            is_best = test_MAE_PER_AVG < BEST_test_MAE_PER_AVG

            if is_best:
                BEST_test_MAE_PER_AVG = min(test_MAE_PER_AVG, BEST_test_MAE_PER_AVG)
                print('BEST_test_MAE_PER_AVG: ', BEST_test_MAE_PER_AVG)

                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'BEST_test_MAE_PER_AVG': BEST_test_MAE_PER_AVG,
                    'optimizer': opt.state_dict()
                }, is_best, epoch + 1)

        writer.add_scalar('train_MAE_PER_AVG', train_MAE_PER_AVG, epoch)
        writer.add_scalar('test_MAE_PER_AVG', test_MAE_PER_AVG, epoch)

    writer.close()
    # 打开Terminal,键入tensorboard --logdir=logs(logs是事件文件所保存路径)


for num in range(0, 10):
    train_once_data(num)
