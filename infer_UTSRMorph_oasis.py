
import glob, sys
import os, losses, utils
import time

from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from models_oasis.UTSRMorph import CONFIGS as CONFIGS_TM
import models_oasis.UTSRMorph as UTSRMorph
from scipy.ndimage.interpolation import map_coordinates, zoom
from surface_distance import *
import scipy.ndimage
def hd95_val_substruct(y_pred, y_true):
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    y_true = y_true[0,0,...]
    y_pred = y_pred[0,0,...]
    hd95 = np.zeros((35))
    idx = 0
    for i in range(1,36):
        if ((y_true == i).sum() == 0) or ((y_pred == i).sum() == 0):
            hd=0
        else:
            hd = compute_robust_hausdorff(compute_surface_distances((y_true==i), (y_pred==i), np.ones(3)), 95.)
        hd95[idx] = hd
        idx = idx+1
    return hd95
def dice_val_VOI(y_pred, y_true):
    VOI_lbls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                29, 30, 31, 32, 33, 34, 35]

    pred = y_pred.detach().cpu().numpy()[0, 0, ...]
    true = y_true.detach().cpu().numpy()[0, 0, ...]
    DSCs = np.zeros((len(VOI_lbls)))
    idx = 0
    for i in VOI_lbls:
        pred_i = pred == i
        true_i = true == i
        intersection = pred_i * true_i
        intersection = np.sum(intersection)
        union = np.sum(pred_i) + np.sum(true_i)
        dsc = (2.*intersection) / (union + 1e-5)
        DSCs[idx] =dsc
        idx += 1
    return DSCs
def jacobian_determinant(disp):
    _, _, H, W, D = disp.shape

    gradx = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)

    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)

    gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (
                jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) - \
             jacobian[1, 0, :, :, :] * (
                         jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :,
                                                                                                       :, :]) + \
             jacobian[2, 0, :, :, :] * (
                         jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :,
                                                                                                       :, :])

    return jacdet
def main():
    test_dir = r'/home/zrs/CTMRI/OASIS_L2R_2021_task03/Test/'
    save_dir = '/home/zrs/CTMRI/OASIS_L2R_2021_task03/Submit/submission/task_03/'
    model_idx = -1
    weights = [1, 1, 1]
    model_folder = 'UTSRMorph_ncc_{}_diffusion_{}/'.format(weights[0], weights[1])
    model_dir = 'experiments/' + model_folder
    config = CONFIGS_TM['UTSRMorph-Small']
    model = UTSRMorph.UTSRMorph(config)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    img_size = (160, 192, 224)
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()
    test_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16)),])
    test_set = datasets.OASISBrainInferDataset(glob.glob(test_dir + '*.pkl'), transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    file_names = glob.glob(test_dir + '*.pkl')
    dice_all = np.zeros([19, 35])
    hd_all = np.zeros([19, 35])
    logj = np.zeros([19])
    with torch.no_grad():
        stdy_idx = 0
        time_start = time.time()
        for data in file_names:
            x, y, x_seg, y_seg = utils.pkload(data)
            x, y = x[None, None, ...], y[None, None, ...]
            x = np.ascontiguousarray(x)
            y = np.ascontiguousarray(y)
            x, y = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()

            x_seg, y_seg = x_seg[None, None, ...], y_seg[None, None, ...]
            x_seg = np.ascontiguousarray(x_seg)
            y_seg = np.ascontiguousarray(y_seg)
            x_seg, y_seg = torch.from_numpy(x_seg).cuda(), torch.from_numpy(y_seg).cuda()
            #a = file_names[stdy_idx].split('/')
            #b=file_names[stdy_idx].split('\\')[-1].split('.')
            file_name = file_names[stdy_idx].split('/')[-1].split('.')[0][2:]
            print(file_name)
            model.eval()
            x_in = torch.cat((x, y),dim=1)
            x_def, flow = model(x_in)
            #flow = flow.cpu().detach().numpy()[0]
            #flow = np.array([zoom(flow[i], 0.5, order=2) for i in range(3)]).astype(np.float16)
            #flow = np.array([zoom(flow.astype(np.float32)[i], 2, order=2) for i in range(3)])[None, ...]

            #def_out = reg_model([x_seg.cuda().float(), torch.from_numpy(flow).cuda()])

            def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
            dsc = dice_val_VOI(def_out.long(), y_seg.long())
            dice_all[stdy_idx, :] = dsc[:]
            hd = hd95_val_substruct(def_out.long(), y_seg.long())
            hd_all[stdy_idx, :] = hd[:]
            flow = flow.cpu().detach().numpy()
            print(flow.shape)
            jac_det = (jacobian_determinant(flow) + 3).clip(0.000000001, 1000000000)
            log_jac_det = np.log(jac_det)
            logj[stdy_idx] = log_jac_det.std()
            print(dsc.shape)
            # print(flow.shape)
            # np.savez(save_dir+'disp_{}.npz'.format(file_name), flow)
            stdy_idx += 1
        print(np.mean(dice_all))
        print(np.std(dice_all))
        print(np.mean(hd_all))
        print(np.std(hd_all))
        print(np.mean(logj))
        print(np.std(logj))
        #np.savetxt('/home/zrs/CTMRI/result/result_utsrmorph_up.txt', dice_all)
        time_end = time.time()
        print(str((time_end - time_start) / 19.0))
if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 1
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