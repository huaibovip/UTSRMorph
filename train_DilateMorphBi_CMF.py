import datetime
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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import losses
import utils_CMF
from data import datasets, trans
from models_dilatemorph import DilateMorphBi


class Logger(object):

    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir + "logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def compterpoint(movingpoint):
    point = np.nonzero(movingpoint)
    pointlacate = []
    pointlabel = []
    for j in range(point[0].shape[0]):
        pointlabel.append(movingpoint[point[0][j], point[1][j], point[2][j]])
        pointlacate.append((point[0][j], point[1][j], point[2][j]))
    pointlacate = np.array(pointlacate)
    #print(pointlabel)
    #print(pointlacate)
    affpoint = np.zeros([9, 3], dtype=np.float32)

    for index in range(pointlabel.__len__()):
        affpoint[int(pointlabel[index]) - 1, :] = pointlacate[index, :]
    return affpoint


def main():
    batch_size = 1
    train_dir = '/root/share/CMF/train_unaff/CT/'
    val_dir = '/root/share/CMF/test_unaff/CT/data/'
    weights = [1, 1, 1]  # loss weights
    save_dir = 'DilateMorphBi_mi{}_diffusion{}/'.format(weights[0], weights[2])
    if not os.path.exists('work_dirs/dilatemorphbi/experiments/' + save_dir):
        os.makedirs('work_dirs/dilatemorphbi/experiments/' + save_dir)
    if not os.path.exists('work_dirs/dilatemorphbi/logs/' + save_dir):
        os.makedirs('work_dirs/dilatemorphbi/logs/' + save_dir)
    sys.stdout = Logger('work_dirs/dilatemorphbi/logs/' + save_dir)
    lr = 0.0001  # learning rate
    epoch_start = 0
    max_epoch = 200  #max traning epoch
    cont_training = False  #if continue training
    '''
    Initialize model
    '''
    img_size = (128, 192, 224)
    model = DilateMorphBi(img_size=img_size,
                          dilation=[2, 3],
                          num_heads=[2, 4, 8, 16],
                          use_checkpoint=True)
    model.cuda()
    '''
    Initialize spatial transformation function
    '''
    reg_model = utils_CMF.Warp(img_size)
    reg_model.cuda()
    '''
    If continue from previous training
    '''
    if cont_training:
        epoch_start = 201
        model_dir = 'experiments/' + save_dir
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
    train_composed = transforms.Compose([
        trans.NumpyType((np.float32, np.int16)),
    ])

    val_composed = transforms.Compose(
        [trans.NumpyType((np.float32, np.int16))])
    train_set = datasets.CMFDataset(glob.glob(train_dir + '*.npy'),
                                    transforms=train_composed)
    val_set = datasets.CMFInferDataset(glob.glob(val_dir + '*.npy'),
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
    criterion_reg = losses.Grad3d(penalty='l2')
    best_dsc = 0
    writer = SummaryWriter(log_dir='work_dirs/dilatemorphbi/logs/' + save_dir)
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        time_start = time.time()
        loss_all = utils_CMF.AverageMeter()
        idx = 0
        for iii in range(15):
            for data in train_loader:
                idx += 1
                model.train()
                adjust_learning_rate(optimizer, epoch, max_epoch, lr)
                data = [t.cuda() for t in data]
                x = data[0]
                y = data[1]

                if True:
                    warped_mov, warped_fix, mov_flow, fix_flow = model(
                        x, y, training=True)
                    loss_sim = 0.5 * (
                        criterion_sim(warped_mov, y) +
                        criterion_sim(warped_fix, x)) * weights[0]
                    loss_reg = 0.5 * (
                        criterion_reg(mov_flow, None) +
                        criterion_reg(fix_flow, None)) * weights[2]
                    del warped_mov, warped_fix, mov_flow, fix_flow
                    loss = loss_sim + loss_reg
                    loss_all.update(loss.item(), y.numel())
                    # compute gradient and do SGD step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                print(
                    'Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.
                    format(idx, 15 * len(train_loader), loss.item(),
                           loss_sim.item(), loss_reg.item()))
        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        '''
        Validation
        '''
        eval_dsc = utils_CMF.AverageMeter()
        with torch.no_grad():
            trepro = []
            pointtre = []

            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data]
                x = data[1]
                y = data[0]
                x_seg = data[3]
                y_seg = data[2]
                grid_img = mk_grid_img(8, 1, img_size)
                output = model(x, y)
                # def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
                def_out = reg_model([x_seg.cuda().float(), output[1].cuda()],
                                    mode='nearest')
                movingfro = x_seg.detach().cpu().numpy()[0, 0, ...]
                fixfro = y_seg.detach().cpu().numpy()[0, 0, ...]
                movingfro = compterpoint(movingfro)
                fixfro = compterpoint(fixfro)

                effectindex = []
                distancepro = []
                for j in range(9):
                    if (movingfro[j, 0] + movingfro[j, 1] + movingfro[j, 2]
                            != 0 and fixfro[j, 0] + fixfro[j, 1] + fixfro[j, 2]
                            != 0):
                        dx = movingfro[j, 0] - fixfro[j, 0]
                        dy = movingfro[j, 1] - fixfro[j, 1]
                        dz = movingfro[j, 2] - fixfro[j, 2]
                        tre = np.sqrt(dx * dx + dy * dy + dz * dz)
                        trepro.append(tre)
                        distancepro.append(tre)
                        effectindex.append(j)

                # def_grid = reg_model_bilin(
                #     [grid_img.float(), output[1].cuda()])
                def_grid = reg_model([grid_img.float(), output[1].cuda()],
                                     mode='bilinear')

                distance = []
                # (160, 192, 224, 3)
                flow = output[1].squeeze().permute(1, 2, 3, 0)
                fix_point = fixfro[np.newaxis, ...]  # (1, 300, 3)
                # 如果要保存的话，可以保存
                dvf_Data = flow.cuda().data.cpu().numpy().squeeze()
                data = [
                    torch.from_numpy(t).cuda() for t in [fix_point, dvf_Data]
                ]  # 由于point_spatial_transformer使用了torch，组装在一起
                warp_point = utils_CMF.point_spatial_transformer(data)

                warp_point = warp_point.cuda().data.cpu().numpy().squeeze()
                for j in effectindex:
                    dx = movingfro[j, 0] - warp_point[j, 0]
                    dy = movingfro[j, 1] - warp_point[j, 1]
                    dz = movingfro[j, 2] - warp_point[j, 2]
                    tre = np.sqrt(dx * dx + dy * dy + dz * dz)
                    pointtre.append(tre)
                    distance.append(tre)
                eval_dsc.update(100 - np.mean(distance), x.size(0))

            print('allpointafter:')
            print(np.mean(pointtre))
            print('alldataforward:')
            print(np.mean(trepro))
            print(eval_dsc.avg)
            best_dsc = min(np.mean(pointtre), best_dsc)
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_dsc': best_dsc,
                'optimizer': optimizer.state_dict(),
            },
            save_dir='work_dirs/dilatemorphbi/experiments/' + save_dir,
            filename='dsc{:.4f}_epoch{:03d}.pth.tar'.format(
                100 - np.mean(pointtre), epoch))

        time_end = time.time()
        alltime = (time_end - time_start) * (199 - epoch)
        timeresult = str(datetime.timedelta(seconds=alltime))
        print("time:" + timeresult)
        writer.add_scalar('DSC/validate', np.mean(pointtre), epoch)
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
                    max_model_num=8):
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
    torch.manual_seed(0)
    main()
    # os.system("/usr/bin/shutdown")
