import glob
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import SimpleITK as sitk
from natsort import natsorted
import pystrum.pynd.ndutils as nd

import utils_abd
from models_dilatemorph import DilateMorph, DilateMorphNoSkip2
from surface_distance import *


def pksave(file_name, target_seg, pred_seg, flow):
    if not os.path.exists('work_dirs/dilatemorph/results'):
        os.makedirs('work_dirs/dilatemorph/results')

    datas = {
        'target_seg': target_seg,
        'pred_seg': pred_seg,
        'pred_flow': flow,
    }
    with open('work_dirs/dilatemorph/results/subject_' + file_name, 'wb') as f:
        pickle.dump(datas, f)


def pksave(file_name, target_seg, pred_seg, flow, mov, fix, pred_img):
    if not os.path.exists('results'):
        os.makedirs('results')
    out = sitk.GetImageFromArray(target_seg.astype('float32'))
    sitk.WriteImage(out, f'results/subject{file_name[:-4]}_fseg.nii.gz')
    out = sitk.GetImageFromArray(pred_seg.astype('float32'))
    sitk.WriteImage(out, f'results/subject{file_name[:-4]}_wseg.nii.gz')
    # out = sitk.GetImageFromArray(flow.astype('float32'))
    # sitk.WriteImage(out, f'results/subject{file_name[:-4]}_flow.nii.gz')
    out = sitk.GetImageFromArray(mov.astype('float32'))
    sitk.WriteImage(out, f'results/subject{file_name[:-4]}_mov.nii.gz')
    out = sitk.GetImageFromArray(fix.astype('float32'))
    sitk.WriteImage(out, f'results/subject{file_name[:-4]}_fix.nii.gz')
    out = sitk.GetImageFromArray(pred_img.astype('float32'))
    sitk.WriteImage(out, f'results/subject{file_name[:-4]}_wmov.nii.gz')


def dice_val_VOI(y_pred, y_true):
    VOI_lbls = [1, 2, 3, 4]
    pred = y_pred.detach().cpu().numpy()[0, 0, ...]
    true = y_true.detach().cpu().numpy()[0, 0, ...]
    DSCs = np.zeros((len(VOI_lbls), 1))
    idx = 0
    for i in VOI_lbls:
        pred_i = pred == i
        true_i = true == i
        intersection = pred_i * true_i
        intersection = np.sum(intersection)
        union = np.sum(pred_i) + np.sum(true_i)
        dsc = (2. * intersection) / (union + 1e-5)
        DSCs[idx] = dsc
        idx += 1
    return DSCs


def hd95_val_substruct(y_pred, y_true, std_idx):
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    y_true = y_true[0, 0, ...]
    y_pred = y_pred[0, 0, ...]
    hd95 = np.zeros((5, 1))
    idx = 0
    for i in range(5):
        if ((y_true == i).sum() == 0) or ((y_pred == i).sum() == 0):
            hd = 0
        else:
            hd = compute_robust_hausdorff(
                compute_surface_distances((y_true == i), (y_pred == i),
                                          np.ones(3)), 95.)
        hd95[idx] = hd
        idx = idx + 1
    return hd95


def jacobian_determinant_vxm(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    disp = disp.transpose(1, 2, 3, 0)
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[...,
                   0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[...,
                   1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[...,
                   2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


def main():
    test_dir = './abdomen/test/CT/data/'
    #save_dir = './result/abdomen/'
    model_idx = -1
    weights = [1, 1, 1]
    # model_folder = 'DilateMorph_mi{}_diffusion{}/'.format(weights[0], weights[2])
    model_folder = 'DilateMorph_mi{}_dsc{}_diffusion{}/'.format(weights[0], weights[1], weights[2])
    model_dir = 'work_dirs/dilatemorph/experiments/' + model_folder
    img_size = (192, 160, 192)
    model = DilateMorphNoSkip2(
        img_size=img_size,
        dilation=[2, 3],
        num_heads=(2, 4, 8, 16),
        use_checkpoint=False,
    )
    best_model = torch.load(
        model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    img_size = (192, 160, 192)
    reg_model = utils_abd.register_model(img_size, 'nearest')
    reg_model.cuda()

    file_names = glob.glob(test_dir + '*.npy')

    with torch.no_grad():
        stdy_idx = 0
        time_start = time.time()
        dice_score = np.zeros([len(file_names), 4])
        hd95 = np.zeros([len(file_names), 5])
        jac = np.zeros([len(file_names), 1])
        for data in file_names:

            file_name = os.path.basename(data)
            x = np.load(data)
            x_seg = np.load(data.replace("data", "mask"))

            y = np.load(data.replace("CT", "MRI"))
            y_seg = np.load(data.replace("CT", "MRI").replace("data", "mask"))
            # x_nii = itk.GetImageFromArray(x.astype(np.float32))
            # x_nii.SetSpacing([2, 2, 2])
            #itk.WriteImage(x_nii, save_dir + file_name.replace(".npy",".nii.gz"))

            # x_nii = itk.GetImageFromArray(y.astype(np.float32))
            # x_nii.SetSpacing([2, 2, 2])
            #itk.WriteImage(x_nii, save_dir.replace("rawct", "rawmr") + file_name.replace(".npy",".nii.gz"))

            # x_nii = itk.GetImageFromArray(x_seg.astype(np.int8))
            # x_nii.SetSpacing([2, 2, 2])
            # itk.WriteImage(x_nii, save_dir.replace("rawct", "rawctseg") + file_name.replace(".npy",".nii.gz"))

            # x_nii = itk.GetImageFromArray(y_seg.astype(np.int8))
            # x_nii.SetSpacing([2, 2, 2])
            # itk.WriteImage(x_nii, save_dir.replace("rawct", "rawmrseg") + file_name.replace(".npy",".nii.gz"))
            #x, y, x_seg, y_seg = utils.pkload(data)
            x, y = x[None, None, ...], y[None, None, ...]

            x_seg, y_seg = x_seg[None, None, ...], y_seg[None, None, ...]
            x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
            y = np.ascontiguousarray(y)
            x_seg = np.ascontiguousarray(
                x_seg)  # [Bsize,channelsHeight,,Width,Depth]
            y_seg = np.ascontiguousarray(y_seg)
            x, y, x_seg, y_seg = torch.from_numpy(x).cuda(), torch.from_numpy(
                y).cuda(), torch.from_numpy(x_seg).cuda(), torch.from_numpy(
                    y_seg).cuda()
            #a = file_names[stdy_idx].split('/')
            #b=file_names[stdy_idx].split('\\')[-1].split('.')
            #file_name = file_names[stdy_idx].split('/')[-1].split('.')[0][2:]
            print(file_name)
            model.eval()
            x_def, flow = model(x, y)

            x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=5)
            x_seg_oh = x_seg_oh.squeeze(1).permute(0, 4, 1, 2, 3).contiguous()
            # x_segs = model.spatial_trans(x_seg_oh.float(), flow.float())
            # dicemean:0.6942510412589162dicestd:0.20176667342205112
            x_segs = []
            for i in range(5):
                def_seg = reg_model(
                    [x_seg_oh[:, i:i + 1, ...].float(),
                     flow.float()])
                x_segs.append(def_seg)
            x_segs = torch.cat(x_segs, dim=1)
            # dicemean:0.6950645906532701dicestd:0.20142164791209782
            def_out = torch.argmax(x_segs, dim=1, keepdim=True)
            pksave(
                file_name,
                y_seg.cpu().numpy(),
                def_out.cpu().numpy(),
                flow.cpu().numpy(),
                x.cpu().numpy(),
                y.cpu().numpy(),
                x_def.cpu().numpy(),
            )
            del x_segs, x_seg_oh
            # def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            jac_det = jacobian_determinant_vxm(
                flow.detach().cpu().numpy()[0, :, :, :, :])
            dsc = dice_val_VOI(def_out.long(), y_seg.long())
            hd = hd95_val_substruct(def_out.long(), y_seg.long(), stdy_idx)
            jac[stdy_idx] = np.sum(jac_det <= 0) / np.prod(tar.shape)
            dice_score[stdy_idx, :] = dsc[:, 0]
            hd95[stdy_idx, :] = hd[:, 0]

            stdy_idx += 1
            #def_out = def_out.long().cpu().detach().numpy()[0, 0, ...].astype(np.int8)
            # x_nii = itk.GetImageFromArray(
            #     def_out.long().cpu().detach().numpy()[0, 0,
            #                                           ...].astype(np.int8))
            # x_nii.SetSpacing([2, 2, 2])
            #itk.WriteImage(x_nii, save_dir.replace("rawct", "resultmrseg") + file_name.replace(".npy",".nii.gz"))
            print(x_def.cpu().detach().numpy()[0, 0,
                                               ...].astype(np.float32).shape)
            # x_nii = itk.GetImageFromArray(
            #     x_def.cpu().detach().numpy()[0, 0, ...].astype(np.float32))
            # x_nii.SetSpacing([2, 2, 2])
            # itk.WriteImage(x_nii, save_dir.replace("rawct", "resultmr") + file_name.replace(".npy",".nii.gz"))

            flow = flow.cpu().detach().numpy()[0]
            flow = np.array([flow[i] for i in range(3)]).astype(np.float16)
            print(flow.shape)
        # np.save(save_dir.replace("rawct", "resultflow") + file_name,flow)

        time_end = time.time()
        print(str((time_end - time_start) / 8.0))
        print("dicemean:" + str(np.mean(dice_score[np.nonzero(dice_score)])) +
              "dicestd:" + str(np.std(dice_score[np.nonzero(dice_score)])))
        print("hd95mean:" + str(np.mean(hd95[np.nonzero(hd95)])) + "hd95std:" +
              str(np.std(hd95[np.nonzero(hd95)])))
        print("jacmean:" + str(np.mean(jac)) + "jacstd:" + str(np.std(jac)))
        #np.savetxt('/home/mh/PythonCodes/xiong_result/result.txt', dice_score, fmt="%f")
        #np.savetxt('/home/mh/PythonCodes/xiong_result/resulthd.txt', hd95, fmt="%f")


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
