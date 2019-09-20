from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader

import glob
from utils.utils import *
import progressbar as pb
import blosc
import multiprocessing
import socket

blosc.set_nthreads(1)


class Pseudo2DDataset(Dataset):

    def __init__(self, dataset_mode='training', dataset_type='normal'):
        root_folder = self.__setup_root_folder()
        assert (root_folder is not None)
        self.num_of_workers = 20

        data_path = os.path.join(root_folder, 'pseudo/pseudo_2D/no_tumor')
        tumor_path = os.path.join(root_folder, 'pseudo/pseudo_2D/tumor')

        image_files = sorted(glob.glob(os.path.join(data_path, '*.nii.gz')))
        if dataset_mode == 'tumor':
            image_files = sorted(glob.glob(os.path.join(tumor_path, '*.nii.gz')))

        print(len(image_files))
        self.dataset_mode = dataset_mode
        num_of_all_files = len(image_files)
        num = num_of_all_files // 10

        if dataset_mode == 'training':
            self.image_files = image_files[4 * num:10 * num]
        elif dataset_mode == 'validation':
            self.image_files = image_files[2 * num:4 * num]
        elif dataset_mode == 'tumor':
            self.image_files = image_files[0:2 * num]
        else:
            raise ValueError('Mode not supported')

        self.atlas_file = os.path.join(root_folder, 'atlas_folder', 'atlas_2D.nii.gz')

        image_io = py_fio.ImageIO()
        self.atlas, _, _, _ = image_io.read_to_nc_format(self.atlas_file, silent_mode=True)

        self.images_dict = None
        self.__multi_threads_loading()

        return

    def __multi_threads_loading(self):
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
        self.images_dict = dict(self.images_dict)
        print("Finish loading images")

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

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_io = py_fio.ImageIO()
        image, _, _, _ = image_io.read_to_nc_format(image_file, silent_mode=True)
        atlas, _, _, _ = image_io.read_to_nc_format(self.atlas_file, silent_mode=True)
        return image[0, ...], atlas[0, ...]

    def __setup_root_folder(self):
        hostname = socket.gethostname()
        root_folder = None
        if hostname == 'biag-gpu0.cs.unc.edu':
            root_folder = '/playpen/xhs400/Research/data/data_for_pregis_net'
        elif hostname == 'biag-w05.cs.unc.edu':
            root_folder = '/playpen1/xhs400/Research/data/data_for_pregis_net'
        return root_folder