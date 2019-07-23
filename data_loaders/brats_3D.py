from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import os
import sys

mermaid_path = "/playpen/xhs400/Research/FPIR/mermaid"
sys.path.append(mermaid_path)
sys.path.append(os.path.join(mermaid_path, 'pyreg'))
sys.path.append(os.path.join(mermaid_path, 'pyreg/libraries'))

import pyreg.fileio as py_fio
import glob


class Brats3DDataset(Dataset):
    def __init__ (self, mode='training'):
        root_folder = '/playpen/xhs400/Research/data/data_for_pregis_net'
        self.mode = mode
        self.data_path = os.path.join(root_folder, 'brats_affined', 'brats_aff_normed')
        images = sorted(glob.glob(os.path.join(self.data_path, '*t1ce.nii.gz')))
        num_of_images = len(images)
        assert(num_of_images == 285) 

        if mode == 'training':
            self.image_files = images[0:171]
        elif mode == 'validation':
            self.image_files = images[171:228]
        elif mode == 'testing':
            self.image_files = images[228:285]
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

























