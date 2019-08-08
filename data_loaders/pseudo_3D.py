from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader

import glob
from utils.utils import *


class Pseudo3DDataset(Dataset):
    def __init__ (self, mode='training', type='normal'):
        root_folder = '/playpen/xhs400/Research/data/data_for_pregis_net'

        oasis_affined_folder = os.path.join(root_folder, 'oasis_affined')
        oasis_brain_folder = os.path.join(oasis_affined_folder, 'oasis_aff_normed')
        #oasis_label_folder = os.path.join(oasis_affined_folder, 'labels')

        atlas_folder = os.path.join(root_folder, 'atlas_folder')
        atlas_file = os.path.join(atlas_folder, 'atlas.nii.gz')

        pseudo_folder = os.path.join(root_folder, 'pseudo_3D')
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


        #print(image_files)

        self.mode = mode
        num_of_all_files = len(image_files)
        num = num_of_all_files // 40

        # split the data for normal training and abnormal training, so that they don't overlap for sanity.
        if type == 'normal':
            if mode == 'training':
                self.image_files = image_files[6*num:20*num]
            elif mode == 'validation':
                self.image_files = image_files[3*num:6*num]
            elif mode == 'testing':
                self.image_files = image_files[0:3*num]
            else:
                raise ValueError('Mode not supported')
        elif type == 'no_tumor' or type == 'tumor':
            if mode == 'training':
                self.image_files = image_files[26*num:40*num]
            elif mode == 'validation':
                self.image_files = image_files[23*num:26*num]
            elif mode == 'testing':
                self.image_files = image_files[20:23*num]
            else:
                raise ValueError('Mode not supported')

        else:
            raise ValueError("Type not supported")

        self.atlas_file = atlas_file
        self.image_io = py_fio.ImageIO()
        self.atlas, _, _, _ = self.image_io.read_to_nc_format(self.atlas_file, silent_mode=True)
        for i in range(len(self.image_files)):
            image, _, _, _ = self.image_io.read_to_nc_format(self.image_files[i], silent_mode=True)
            print("Image {}: {} is loaded".format(i+1, self.image_files[i]))
            if i == 0:
                self.images = image
            else:
                self.images = np.concatenate((self.images, image), axis=0)
        print(self.images.shape)


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        return self.images[idx,...], self.atlas[0,...]

























