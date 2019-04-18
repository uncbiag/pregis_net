from __future__ import print_function, division
import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys
import numpy as np

mermaid_path = "/playpen/xhs400/Research/FPIR/mermaid"
sys.path.append(mermaid_path)
sys.path.append(os.path.join(mermaid_path, 'pyreg'))
sys.path.append(os.path.join(mermaid_path, 'pyreg/libraries'))

import pyreg.fileio as py_fio
import glob


class Pseudo2DDataset(Dataset):
    def __init__ (self, mode='training'):
        self.mode = mode
        self.data_path = os.path.join(os.path.dirname(__file__), '../../data_2d_syn/tumor_test')
#        self.no_tumor_data_path = os.path.join(os.path.dirname(__file__), '../../data_2d_syn/no_tumor_test')
        
        if mode == 'training':
            self.image_files = sorted(glob.glob(os.path.join(self.data_path, '*.nii.gz')))[20:100]
        elif mode == 'validation':
            self.image_files = sorted(glob.glob(os.path.join(self.data_path, '*.nii.gz')))[0:20]
        elif mode == 'testing':
            self.image_files = sorted(glob.glob(os.path.join(self.data_path, '*.nii.gz')))[80:100]
        else:
            raise ValueError('Mode Wrong!')

        self.atlas_file = os.path.join(self.data_path, '../atlas_2D.nii.gz')
    def __len__(self):
        return len(self.image_files)


    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_io = py_fio.ImageIO()
        image, _, _, _ = image_io.read_to_nc_format(image_file, silent_mode=True)
        atlas, _, _, _ = image_io.read_to_nc_format(self.atlas_file, silent_mode=True)
        return image[0,...], atlas[0,...]


























