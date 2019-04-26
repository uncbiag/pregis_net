from __future__ import print_function, division
import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys
import numpy as np

from utils.utils import *
import glob


class Pseudo2DDataset(Dataset):

    def __init__ (self, mode='training'):
        root_folder = '/playpen/xhs400/Research/data/data_for_pregis_net' 
        data_path = os.path.join(root_folder, 'pseudo_2D')
        
        if mode == 'training':
            self.image_files = sorted(glob.glob(os.path.join(data_path, '*.nii.gz')))[20:100]
        elif mode == 'validation':
            self.image_files = sorted(glob.glob(os.path.join(data_path, '*.nii.gz')))[0:20]
        elif mode == 'testing':
            self.image_files = sorted(glob.glob(os.path.join(data_path, '*.nii.gz')))[0:100]
        else:
            raise ValueError('Mode Wrong!')

        self.atlas_file = os.path.join(root_folder, 'atlas_folder', 'atlas_2D.nii.gz')


    def __len__(self):
        return len(self.image_files)


    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_io = py_fio.ImageIO()
        image, _, _, _ = image_io.read_to_nc_format(image_file, silent_mode=True)
        atlas, _, _, _ = image_io.read_to_nc_format(self.atlas_file, silent_mode=True)
        return image[0,...], atlas[0,...]


























