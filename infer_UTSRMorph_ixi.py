import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn
from natsort import natsorted
from torch.utils.data import DataLoader
from torchvision import transforms

import losses
import models_ixi.UTSRMorph as UTSRMorph
import utils_ixi
from data import datasets, trans
from models_ixi.UTSRMorph import CONFIGS as CONFIGS_UM


def main():
    atlas_dir = '/home/buaaa302/pythoncodes/TransFrame/IXI_data/atlas.pkl'
    test_dir = '/home/buaaa302/pythoncodes/TransFrame/IXI_data/Test/'
    model_idx = -1
    weights = [1, 4]
    model_folder = 'UTSRMorph_ncc_{}_diffusion_{}/'.format(
        weights[0], weights[1])
    model_dir = 'experiments/' + model_folder
    if 'Val' in test_dir:
        csv_name = model_folder[:-1] + '_Val'
    else:
        csv_name = model_folder[:-1]
    dict = utils_ixi.process_label()
    if not os.path.exists('Quantitative_Results/'):
        os.makedirs('Quantitative_Results/')
    if os.path.exists('Quantitative_Results/' + csv_name + '.csv'):
        os.remove('Quantitative_Results/' + csv_name + '.csv')
    csv_writter(model_folder[:-1], 'Quantitative_Results/' + csv_name)
    line = ''
    for i in range(46):
        line = line + ',' + dict[i]
    csv_writter(line + ',' + 'non_jec', 'Quantitative_Results/' + csv_name)

    config = CONFIGS_UM['UTSRMorph']
    model = UTSRMorph.UTSRMorph(config)
    best_model = torch.load(
        model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils_ixi.register_model(config.img_size, 'bilinear')
    reg_model.cuda()
    test_composed = transforms.Compose([
        trans.Seg_norm(),
        trans.NumpyType((np.float32, np.int16)),
    ])
    test_set = datasets.IXIBrainInferDataset(glob.glob(test_dir + '*.pkl'),
                                             atlas_dir,
                                             transforms=test_composed)
    test_loader = DataLoader(test_set,
                             batch_size=1,
                             shuffle=False,
                             num_workers=1,
                             pin_memory=True,
                             drop_last=True)
    eval_dsc_def = utils_ixi.AverageMeter()
    eval_dsc_raw = utils_ixi.AverageMeter()
    eval_det = utils_ixi.AverageMeter()

    start_time = time.time()
    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]

            x_in = torch.cat((x, y), dim=1)
            x_def, flow = model(x_in)
    end_time = time.time()
    print(start_time)
    print(end_time)
    print(str((end_time - start_time) / 115.0))

    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]

            x_in = torch.cat((x, y), dim=1)
            x_def, flow = model(x_in)

            x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=46)
            x_seg_oh = torch.squeeze(x_seg_oh, 1)
            x_seg_oh = x_seg_oh.permute(0, 4, 1, 2, 3).contiguous()
            #x_segs = model.spatial_trans(x_seg.float(), flow.float())
            x_segs = []
            for i in range(46):
                def_seg = reg_model(
                    [x_seg_oh[:, i:i + 1, ...].float(),
                     flow.float()])
                x_segs.append(def_seg)
            x_segs = torch.cat(x_segs, dim=1)
            def_out = torch.argmax(x_segs, dim=1, keepdim=True)
            del x_segs, x_seg_oh
            #def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            jac_det = utils_ixi.jacobian_determinant_vxm(
                flow.detach().cpu().numpy()[0, :, :, :, :])
            #line = utils_ixi.dice_val_substruct(def_out.long(), y_seg.long(), stdy_idx)
            line = utils_ixi.hd95_val_substruct(def_out.long(), y_seg.long(),
                                                stdy_idx)
            line = line + ',' + str(np.sum(jac_det <= 0) / np.prod(tar.shape))
            csv_writter(line, 'Quantitative_Results/' + csv_name)
            eval_det.update(
                np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))
            print('det < 0: {}'.format(
                np.sum(jac_det <= 0) / np.prod(tar.shape)))

            dsc_trans = utils_ixi.dice_val(def_out.long(), y_seg.long(), 46)
            dsc_raw = utils_ixi.dice_val(x_seg.long(), y_seg.long(), 46)
            print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(
                dsc_trans.item(), dsc_raw.item()))
            eval_dsc_def.update(dsc_trans.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            stdy_idx += 1

        print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.
              format(eval_dsc_def.avg, eval_dsc_def.std, eval_dsc_raw.avg,
                     eval_dsc_raw.std))
        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))


def csv_writter(line, name):
    with open(name + '.csv', 'a') as file:
        file.write(line)
        file.write('\n')


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
