
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

def main():
    test_dir = r'/home/zrs/CTMRI/OASIS_L2R_2021_task03/Test/'
    save_dir = '/home/zrs/CTMRI/OASIS_L2R_2021_task03/Submit/submission/task_03/'
    model_idx = -1
    weights = [1, 1, 1]
    model_folder = 'TransMorph_ncc_{}_dsc{}_diffusion_{}/'.format(weights[0], weights[1], weights[2])
    model_dir = 'experiments/' + model_folder
    config = CONFIGS_TM['UTSRMorph']
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
    with torch.no_grad():
        stdy_idx = 0
        time_start = time.time()
        for data in file_names:
            x, y, x_seg, y_seg = utils.pkload(data)
            x, y = x[None, None, ...], y[None, None, ...]
            x = np.ascontiguousarray(x)
            y = np.ascontiguousarray(y)
            x, y = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()
            #a = file_names[stdy_idx].split('/')
            #b=file_names[stdy_idx].split('\\')[-1].split('.')
            file_name = file_names[stdy_idx].split('/')[-1].split('.')[0][2:]
            print(file_name)
            model.eval()
            x_in = torch.cat((x, y),dim=1)
            x_def, flow = model(x_in)
            flow = flow.cpu().detach().numpy()[0]
            flow = np.array([zoom(flow[i], 0.5, order=2) for i in range(3)]).astype(np.float16)
            print(flow.shape)
            np.savez(save_dir+'disp_{}.npz'.format(file_name), flow)
            stdy_idx += 1
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