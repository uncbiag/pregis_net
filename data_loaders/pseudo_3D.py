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


class Pseudo3DDataset(Dataset):
    def __init__ (self, mode='training'):
        self.mode = mode
        root_folder = '/playpen/xhs400/Research/data/data_for_pregis_net'
        self.data_path = os.path.join(root_folder, 'pseudo_3D')  
        #images = sorted(glob.glob(os.path.join(self.data_path, '*t1ce.nii.gz')))
        #num_of_images = len(images)
        #assert(num_of_images == 285) 

        if mode == 'training':
            self.image_path = os.path.join(self.data_path, 'train')
            self.mode = 'train'
        elif mode == 'validation':
            self.image_path = os.path.join(self.data_path, 'validate')
            self.mode = 'validate'
        elif mode == 'testing':
            self.image_path = os.path.join(self.data_path, 'test')
            self.mode = 'test'
        else:
            raise ValueError('Mode Wrong!')
        self.tumor_path = os.path.join(self.image_path, 'tumor_' + self.mode)
        self.no_tumor_path = os.path.join(self.image_path, 'no_tumor_' + self.mode)
        self.mask_path = os.path.join(self.image_path, 'mask_' + self.mode)
        self.tumor_files = sorted(glob.glob(os.path.join(self.tumor_path, '*.nii.gz')))
        self.no_tumor_files = sorted(glob.glob(os.path.join(self.no_tumor_path, '*.nii.gz')))
        self.mask_files = sorted(glob.glob(os.path.join(self.mask_path, '*.nii.gz')))

        self.atlas_file = os.path.join(root_folder, 'atlas_folder', 'atlas.nii.gz')



    def __len__(self):
        return 1
        #return len(self.image_files)


    def __getitem__(self, idx):
        tumor_file = self.tumor_files[idx]
        no_tumor_file = self.no_tumor_files[idx]
        mask_file = self.mask_files[idx]

        image_io = py_fio.ImageIO()
        tumor_image, _, _, _ = image_io.read_to_nc_format(tumor_file, silent_mode=True)
        no_tumor_image, _, _, _ = image_io.read_to_nc_format(no_tumor_file, silent_mode=True)
        mask_image, _, _, _ = image_io.read_to_nc_format(mask_file, silent_mode=True)
        atlas, _, _, _ = image_io.read_to_nc_format(self.atlas_file, silent_mode=True)
        return tumor_image[0,...], atlas[0,...]

























