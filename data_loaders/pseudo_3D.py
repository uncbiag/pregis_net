from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader

import numpy as np
import glob
from utils.utils import *
import progressbar as pb
import blosc
import multiprocessing
import socket
blosc.set_nthreads(1)


class Pseudo3DDataset(Dataset):
    def __init__(self, dataset_mode='training', network_mode='pregis'):

        root_folder = self.__set_root_folder()
        assert(root_folder is not None)
        self.num_of_workers = 12
        oasis_affined_folder = os.path.join(root_folder, 'oasis_3', 'affined')
        oasis_brain_folder = os.path.join(oasis_affined_folder, 'normalized', 'pseudo_not_used')

        atlas_folder = os.path.join(root_folder, 'atlas_folder')
        atlas_file = os.path.join(atlas_folder, 'atlas.nii.gz')

        pseudo_folder = os.path.join(root_folder, 'pseudo', 'pseudo_3D')
        pseudo_tumor_folder = os.path.join(pseudo_folder, 'tumor')
        # pseudo_no_tumor_folder = os.path.join(pseudo_folder, 'no_tumor')

        print("Network Mode: {}".format(network_mode))
        if network_mode == 'pregis':
            image_files = sorted(glob.glob(os.path.join(pseudo_tumor_folder, '*.nii.gz')))
        elif network_mode == 'mermaid' or network_mode == 'recons':
            image_files = sorted(glob.glob(os.path.join(oasis_brain_folder, '*.nii.gz')))
        else:
            raise ValueError("Network mode not supported")

        self.dataset_mode = dataset_mode
        num_of_all_files = len(image_files)
        num = num_of_all_files // 10

        if dataset_mode == 'training':
            self.image_files = image_files[4 * num:10 * num]
        elif dataset_mode == 'validation':
            self.image_files = image_files[2 * num:4 * num]
        elif dataset_mode == 'testing':
            self.image_files = image_files[0:2 * num]
        else:
            raise ValueError('Mode not supported')

        self.atlas_file = atlas_file
        image_io = py_fio.ImageIO()
        self.atlas, _, _, _ = image_io.read_to_nc_format(self.atlas_file, silent_mode=True)

        manager = multiprocessing.Manager()
        self.images_dict = manager.dict()
        split_list = self.__split_image_files()
        process = []
        for i in range(self.num_of_workers):
            p = multiprocessing.Process(target=self.__load_images_and_compress, args=(split_list[i],))
            p.start()
            process.append(p)

        for p in process:
            p.join()
        print("Finish loading images")
        return

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_compressed = self.images_dict[image_name]
        image = blosc.unpack_array(image_compressed)
        return image[0, ...], self.atlas[0, ...]

    def __split_image_files(self):
        index_list = list(range(len(self.image_files)))
        index_split = np.array_split(np.array(index_list), self.num_of_workers)
        split_list = []
        for i in range(self.num_of_workers):
            current_list = self.image_files[index_split[i][0]:index_split[i][0] + len(index_split[i])]
            split_list.append(current_list)
        return split_list

    def __load_images_and_compress(self, image_list):
        pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(image_list)).start()
        count = 0
        image_io = py_fio.ImageIO()

        for fn in image_list:
            image, _, _, _ = image_io.read_to_nc_format(fn, silent_mode=True)
            image_compressed = blosc.pack_array(image)
            self.images_dict[fn] = image_compressed
            count += 1
            pbar.update(count)
        pbar.finish()


    def __set_root_folder(self):
        hostname = socket.gethostname()
        root_folder = None
        if hostname == 'biag-gpu0.cs.unc.edu':
            root_folder = '/playpen/xhs400/Research/data/data_for_pregis_net'
        elif hostname == 'biag-w05.cs.unc.edu':
            root_folder = '/playpen1/xhs400/Research/data/data_for_pregis_net'
        return root_folder
