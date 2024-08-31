
import glob, sys
import os, losses, utils_CMF
import time

from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from models_CMF.UTSRMorph import CONFIGS as CONFIGS_TM
import models_CMF.UTSRMorph as UTSRMorph
from scipy.ndimage.interpolation import map_coordinates, zoom
#from models_rdp import RDP
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
    test_dir = r'/home/buaaa302/pythoncodes/TransFrame/plapairsegaff/test_unaff/CT/data/'
    save_dir = '/home/buaaa302/pythoncodes/cmfresult/rdp/'
    model_idx = -1
    weights = [1, 1, 1]
    model_folder = 'UTSRMorph_ncc_{}_dsc{}_diffusion_{}/'.format(weights[0], weights[1], weights[2])
    model_dir = 'experiments/' + model_folder
    img_size = (128, 192, 224)
    config = CONFIGS_TM['UTSRMorph']
    model = UTSRMorph.UTSRMorph(config)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils_CMF.register_model(img_size, 'nearest')
    reg_model.cuda()
    test_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16)),])
    test_set = datasets.CMFInferDataset(glob.glob(test_dir + '*.npy'), transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    file_names = glob.glob(test_dir + '*.npy')
    with torch.no_grad():
        stdy_idx = 0
        time_start = time.time()
        trepro = []
        pointtre = []
        premovingpoint = np.zeros([63, 3])
        prefropoint = np.zeros([63, 3])
        preerror = np.zeros([63, 1])
        aftermovingpoint = np.zeros([63, 3])
        afterfropoint = np.zeros([63, 3])
        aftererror = np.zeros([63, 1])
        eval_det = utils_CMF.AverageMeter()
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[1]
            y = data[0]
            x_seg = data[3]
            y_seg = data[2]
            output = model(torch.cat((x, y), dim=1))
            def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
            movingfro = x_seg.detach().cpu().numpy()[0, 0, ...]
            fixfro = y_seg.detach().cpu().numpy()[0, 0, ...]
            movingfro = compterpoint(movingfro)
            fixfro = compterpoint(fixfro)

            #np.savetxt(save_dir + data[-12:].replace(".npy",".txt"), movingfro)
            effectindex = []
            distancepro = []
            for j in range(9):
                if (movingfro[j, 0] + movingfro[j, 1] + movingfro[j, 2] != 0 and fixfro[j, 0] + fixfro[j, 1] + fixfro[
                    j, 2] != 0):
                    dx = movingfro[j, 0] - fixfro[j, 0]
                    dy = movingfro[j, 1] - fixfro[j, 1]
                    dz = movingfro[j, 2] - fixfro[j, 2]
                    tre = np.sqrt(dx * dx + dy * dy + dz * dz)
                    trepro.append(tre)
                    distancepro.append(tre)
                    effectindex.append(j)
                    preerror[stdy_idx * 9 + j, 0] = tre

            flow = output[1].squeeze().permute(1, 2, 3, 0)  # (160, 192, 224, 3)
            fix_point = fixfro[np.newaxis, ...]  # (1, 300, 3)
            dvf_Data = flow.cuda().data.cpu().numpy().squeeze()  # 如果要保存的话，可以保存
            data = [torch.from_numpy(t).cuda() for t in
                    [fix_point, dvf_Data]]  # 由于point_spatial_transformer使用了torch，组装在一起
            warp_point = utils_CMF.point_spatial_transformer(data)

            warp_point = warp_point.cuda().data.cpu().numpy().squeeze()
            for j in effectindex:
                dx = movingfro[j, 0] - warp_point[j, 0]
                dy = movingfro[j, 1] - warp_point[j, 1]
                dz = movingfro[j, 2] - warp_point[j, 2]
                tre = np.sqrt(dx * dx + dy * dy + dz * dz)
                pointtre.append(tre)
                aftererror[stdy_idx * 9 + j,0] = tre
            stdy_idx += 1
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            jac_det = utils_CMF.jacobian_determinant_vxm(output[1].detach().cpu().numpy()[0, :, :, :, :])
            eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))
            print('det < 0: {}'.format(np.sum(jac_det <= 0) / np.prod(tar.shape)))
        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))
        print('propoint error: {}, std: {}'.format(np.mean(trepro), np.std(trepro)))
        print('propoint error: {}, std: {}'.format(np.mean(pointtre), np.std(pointtre)))
        np.savetxt(save_dir + "utsrmorph_pre.txt", preerror,fmt='%.3f')
        np.savetxt(save_dir + "utsrmorph_after.txt", aftererror,fmt='%.3f')
        time_end = time.time()
        print(str((time_end - time_start) / 4.0))
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
    main()