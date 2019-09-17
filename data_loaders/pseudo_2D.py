from __future__ import print_function, division
from torch.utils.data import Dataset

import numpy as np
import glob
from utils.utils import *


class Pseudo2DDataset(Dataset):

    def __init__(self, dataset_mode='training', dataset_type='normal'):
        root_folder = '/playpen1/xhs400/Research/data/data_for_pregis_net'
        data_path = os.path.join(root_folder, 'data_2d_syn/no_tumor_test')
        tumor_path = os.path.join(root_folder, 'data_2d_syn/tumor_test')

        if dataset_mode == 'training':
            self.image_files = sorted(glob.glob(os.path.join(data_path, '*.nii.gz')))[0:80]
        elif dataset_mode == 'validation':
            self.image_files = sorted(glob.glob(os.path.join(data_path, '*.nii.gz')))[80:100]
        elif dataset_mode == 'tumor':
            self.image_files = sorted(glob.glob(os.path.join(tumor_path, '*.nii.gz')))[90:100]
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
        return image[0, ...], atlas[0, ...]
