from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader

import glob
from utils.utils import *


class Pseudo3DDataset(Dataset):
    def __init__ (self, mode='training', type='normal'):
        root_folder = '/playpen/xhs400/Research/data/data_for_pregis_net'

        oasis_affined_folder = os.path.join(root_folder, 'oasis_3', 'affined')
        oasis_brain_folder = os.path.join(oasis_affined_folder, 'normalized', 'pseudo_not_used')

        atlas_folder = os.path.join(root_folder, 'atlas_folder')
        atlas_file = os.path.join(atlas_folder, 'atlas.nii.gz')

        pseudo_folder = os.path.join(root_folder, 'pseudo', 'pseudo_3D')
        pseudo_tumor_folder = os.path.join(pseudo_folder, 'tumor')
        pseudo_no_tumor_folder = os.path.join(pseudo_folder, 'no_tumor')

        print(type)
        if type == 'tumor' :
            image_files = sorted(glob.glob(os.path.join(pseudo_tumor_folder, '*.nii.gz')))
        elif type == 'no_tumor' : # These are not normal images
            image_files = sorted(glob.glob(os.path.join(pseudo_no_tumor_folder, '*.nii.gz')))
        elif type == 'normal':
            image_files = sorted(glob.glob(os.path.join(oasis_brain_folder, '*.nii.gz')))
        else:
            raise ValueError("Type not supported")


        self.mode = mode
        num_of_all_files = len(image_files)
        num = num_of_all_files // 10

        # split the data for normal training and abnormal training, so that they don't overlap for sanity.
        if mode == 'training':
            self.image_files = image_files[4*num:10*num]
        elif mode == 'validation':
            self.image_files = image_files[2*num:4*num]
        elif mode == 'testing':
            self.image_files = image_files[0:2*num]
        else:
            raise ValueError('Mode not supported')


        self.atlas_file = atlas_file
        image_io = py_fio.ImageIO()
        self.atlas, _, _, _ = image_io.read_to_nc_format(self.atlas_file, silent_mode=True)
        image, _, _, _ = image_io.read_to_nc_format(self.image_files[0], silent_mode=True)

        self.images, _, _, _ = image_io.read_batch_to_nc_format(self.image_files, silent_mode=True)


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        return self.images[idx,...], self.atlas[0,...]

























