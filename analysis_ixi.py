import numpy as np
import csv, sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from scipy.stats import wilcoxon, ttest_rel, ttest_ind

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    for patch in bp['boxes']:
        patch.set(facecolor = color)
    plt.setp(bp['whiskers'], color='cornflowerblue')
    plt.setp(bp['caps'], color='steelblue')
    plt.setp(bp['medians'], color='dodgerblue')

file_dir = 'Quantitative_Results/'
file_name = ['UTSRMorph_dice_test']
substruct = ['Left-Cerebral-White-Matter','Left-Cerebral-Cortex','Left-Lateral-Ventricle','Left-Inf-Lat-Vent','Left-Cerebellum-White-Matter','Left-Cerebellum-Cortex','Left-Thalamus-Proper*',
             'Left-Caudate','Left-Putamen','Left-Pallidum','3rd-Ventricle','4th-Ventricle','Brain-Stem','Left-Hippocampus','Left-Amygdala','CSF','Left-Accumbens-area','Left-VentralDC',
             'Left-vessel','Left-choroid-plexus','Right-Cerebral-White-Matter','Right-Cerebral-Cortex','Right-Lateral-Ventricle','Right-Inf-Lat-Vent','Right-Cerebellum-White-Matter',
             'Right-Cerebellum-Cortex','Right-Thalamus-Proper*','Right-Caudate','Right-Putamen','Right-Pallidum','Right-Hippocampus','Right-Amygdala','Right-Accumbens-area','Right-VentralDC',
             'Right-vessel','Right-choroid-plexus','5th-Ventricle','WM-hypointensities','non-WM-hypointensities','Optic-Chiasm','CC_Posterior','CC_Mid_Posterior','CC_Central','CC_Mid_Anterior,CC_Anterior']

outstruct = ['Brain-Stem', 'Thalamus', 'Cerebellum-Cortex', 'Cerebral-White-Matter', 'Cerebellum-White-Matter', 'Putamen', 'VentralDC', 'Pallidum', 'Caudate', 'Lateral-Ventricle', 'Hippocampus',
             '3rd-Ventricle', '4th-Ventricle', 'Amygdala', 'Cerebral-Cortex', 'CSF', 'choroid-plexus']
all_data = []
all_dsc = []
for exp_name in file_name:
    print(exp_name)
    exp_data = np.zeros((len(outstruct), 115))
    stct_i = 0
    for stct in outstruct:
        tar_idx = []
        with open(file_dir+exp_name+'.csv', "r") as f:
            reader = csv.reader(f, delimiter="\t")
            for i, line in enumerate(reader):
                if i == 1:
                    names = line[0].split(',')
                    idx = 0
                    for item in names:
                        if stct in item:
                            tar_idx.append(idx)
                        idx += 1
                elif i>1:
                    if line[0].split(',')[1]=='':
                        continue
                    val = 0
                    for lr_i in tar_idx:
                        vals = line[0].split(',')
                        val += float(vals[lr_i])
                    val = val/len(tar_idx)
                    exp_data[stct_i, i-2] = val
                    #print(stct_i)
        stct_i+=1
    all_dsc.append(exp_data.mean(axis=0))
    print(exp_data.mean())
    print(exp_data.std())
    all_data.append(exp_data)
    my_list = []
    with open(file_dir + exp_name + '.csv', newline='') as f:
        reader = csv.reader(f)
        my_list = [row[-1] for row in reader]
    my_list = my_list[2:]
    my_list = np.array([float(i) for i in my_list])*100
    print('jec_det: {:.3f} +- {:.3f}'.format(my_list.mean(), my_list.std()))