from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader

import glob
from utils.utils import *


class Pseudo3DDataset(Dataset):
    def __init__ (self, mode='training'):
        root_folder = '/playpen/xhs400/Research/data/data_for_pregis_net'
        data_path = os.path.join(root_folder, 'pseudo_3D', 'tumor')  
        image_files = sorted(glob.glob(os.path.join(data_path, '*.nii.gz')))
        num_of_all_files = len(image_files)
        num = num_of_all_files//5
        self.mode = mode
        if mode == 'training':
            self.image_files = image_files[2*num:]
        elif mode == 'validation':
            self.image_files = image_files[1*num:2*num]
        elif mode == 'testing':
            self.image_files = image_files[0:1*num]
        else:
            raise ValueError('Mode Wrong!')
        self.atlas_file = os.path.join(root_folder, 'atlas_folder', 'atlas.nii.gz')

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_io = py_fio.ImageIO()
        image, _, _, _ = image_io.read_to_nc_format(image_file, silent_mode=True)
        atlas, _, _, _ = image_io.read_to_nc_format(self.atlas_file, silent_mode=True)
        return image[0,...], atlas[0,...]

























