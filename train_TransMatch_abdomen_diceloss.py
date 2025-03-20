import datetime
import gc
import glob
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from natsort import natsorted
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import losses
import utils_abd
from data import datasets, trans
from models_transmatch import TransMatch


def main():
    batch_size = 1
    train_dir = 'abdomen/train/CT/data/'
    val_dir = 'abdomen/test/CT/data/'
    weights = [1, 1, 4]  # loss weights
    save_dir = 'TransMatch_mi{}_dsc{}_diffusion{}/'.format(
        weights[0], weights[1], weights[2])
    if not os.path.exists('work_dirs/transmatch/experiments/' + save_dir):
        os.makedirs('work_dirs/transmatch/experiments/' + save_dir)
    if not os.path.exists('work_dirs/transmatch/logs/' + save_dir):
        os.makedirs('work_dirs/transmatch/logs/' + save_dir)
    sys.stdout = Logger('work_dirs/transmatch/logs/' + save_dir)
    lr = 0.0004  # learning rate
    epoch_start = 0
    max_epoch = 500  #max traning epoch
    cont_training = False  #if continue training
    '''
    Initialize model
    '''
    img_size = (192, 160, 192)
    model = TransMatch(
        img_size=img_size,
        window_size=[6, 5, 6],
        # use_checkpoint=True,
    )
    model.cuda()
    '''
    Initialize spatial transformation function
    '''
    reg_model = utils_abd.Warp(img_size)
    reg_model.cuda()
    '''
    If continue from previous training
    '''
    if cont_training:
        epoch_start = 180
        model_dir = 'work_dirs/transmatch/experiments/' + save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch, 0.9),
                           8)
        best_model = torch.load(
            model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
        print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[-1]))
        model.load_state_dict(best_model)
    else:
        updated_lr = lr
    '''
    Initialize training
    '''
    train_composed = transforms.Compose(
        [trans.NumpyType((np.float32, np.int16))])
    val_composed = transforms.Compose(
        [trans.NumpyType((np.float32, np.int16))])
    train_set = datasets.CTMRIABDDataset(glob.glob(train_dir + '*.npy'),
                                         transforms=train_composed)
    val_set = datasets.CTMRIABDInferDataset(glob.glob(val_dir + '*.npy'),
                                            transforms=val_composed)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(val_set,
                            batch_size=1,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=True)

    optimizer = optim.Adam(model.parameters(),
                           lr=updated_lr,
                           weight_decay=0,
                           amsgrad=True)
    criterion_sim = losses.MutualInformation()
    criterion_dsc = losses.DiceLoss(num_class=5)
    criterion_reg = losses.Grad3d(penalty='l2')
    best_dsc = 0
    writer = SummaryWriter(log_dir='work_dirs/transmatch/logs/' + save_dir)
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils_abd.AverageMeter()
        idx = 0
        time_start = time.time()
        for ii in range(8):
            for data in train_loader:
                idx += 1
                model.train()
                adjust_learning_rate(optimizer, epoch, max_epoch, lr)
                data = [t.cuda() for t in data]
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]

                if (random.random() > 0.5):
                    x_seg_oh = F.one_hot(x_seg.long(), num_classes=5)
                    x_seg_oh = torch.squeeze(x_seg_oh, 1)
                    x_seg_oh = x_seg_oh.permute(0, 4, 1, 2, 3).contiguous()

                    output, flow = model(x, y)
                    def_segs = []
                    for i in range(5):
                        def_seg = model.warp(x_seg_oh[:, i:i + 1, ...].float(),
                                             flow=flow.float())
                        def_segs.append(def_seg)
                    def_seg = torch.cat(def_segs, dim=1)
                    loss_ncc = criterion_sim(output, y) * weights[0]
                    loss_reg = criterion_reg(flow, y) * weights[2]
                    del def_segs, output, flow, x_seg_oh,
                    loss_dsc = criterion_dsc(def_seg,
                                             y_seg.long()) * weights[1]

                    loss = loss_ncc + loss_reg + loss_dsc
                    loss_all.update(loss.item(), y.numel())
                    # compute gradient and do SGD step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    del def_seg
                else:
                    y_seg_oh = F.one_hot(y_seg.long(), num_classes=5)
                    y_seg_oh = torch.squeeze(y_seg_oh, 1)
                    y_seg_oh = y_seg_oh.permute(0, 4, 1, 2, 3).contiguous()

                    output, flow = model(y, x)
                    def_segs = []
                    for i in range(5):
                        def_seg = model.warp(y_seg_oh[:, i:i + 1, ...].float(),
                                             flow=flow.float())
                        def_segs.append(def_seg)
                    def_seg = torch.cat(def_segs, dim=1)
                    loss_ncc = criterion_sim(output, x) * weights[0]
                    loss_reg = criterion_reg(flow, x) * weights[2]
                    del def_segs, output, flow, y_seg_oh,
                    loss_dsc = criterion_dsc(def_seg,
                                             x_seg.long()) * weights[1]
                    loss = loss_ncc + loss_reg + loss_dsc
                    loss_all.update(loss.item(), x.numel())
                    # compute gradient and do SGD step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    del def_seg

                print(
                    'Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Dsc: {:.6f}, Reg: {:.6f}'
                    .format(idx,
                            len(train_loader) * 8, loss.item(),
                            loss_ncc.item(), loss_dsc.item(), loss_reg.item()))
                del loss, loss_dsc, loss_ncc, loss_reg

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        torch.cuda.empty_cache()
        gc.collect()
        '''
        Validation
        '''
        eval_dsc = utils_abd.AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data]
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]
                grid_img = mk_grid_img(8, 1, img_size)
                output = model(x, y)
                def_out = reg_model([x_seg.cuda().float(), output[1].cuda()],
                                    mode='nearest')
                def_grid = reg_model([grid_img.float(), output[1].cuda()],
                                     mode='bilinear')
                dsc = utils_abd.dice_val_VOI(def_out.long(), y_seg.long())
                eval_dsc.update(dsc.item(), x.size(0))
                print(eval_dsc.avg)
        best_dsc = max(eval_dsc.avg, best_dsc)
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_dsc': best_dsc,
                'optimizer': optimizer.state_dict(),
            },
            save_dir='work_dirs/transmatch/experiments/' + save_dir,
            filename='dsc{:.4f}_epoch{:03d}.pth.tar'.format(
                eval_dsc.avg, epoch))
        time_end = time.time()
        alltime = (time_end - time_start) * (499 - epoch)
        timeresult = str(datetime.timedelta(seconds=alltime))
        print("time:" + timeresult)
        writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)
        plt.switch_backend('agg')
        pred_fig = comput_fig(def_out)
        grid_fig = comput_fig(def_grid)
        x_fig = comput_fig(x_seg)
        tar_fig = comput_fig(y_seg)
        writer.add_figure('Grid', grid_fig, epoch)
        plt.close(grid_fig)
        writer.add_figure('input', x_fig, epoch)
        plt.close(x_fig)
        writer.add_figure('ground truth', tar_fig, epoch)
        plt.close(tar_fig)
        writer.add_figure('prediction', pred_fig, epoch)
        plt.close(pred_fig)
        loss_all.reset()
        del def_out, def_grid, grid_img, output
    writer.close()


class Logger(object):

    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir + "logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(
            INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j + line_thickness - 1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i + line_thickness - 1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img


def save_checkpoint(state,
                    save_dir='models',
                    filename='checkpoint.pth.tar',
                    max_model_num=3):
    torch.save(state, save_dir + filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))


if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    # torch.manual_seed(0)
    main()
    # os.system("/usr/bin/shutdown")
